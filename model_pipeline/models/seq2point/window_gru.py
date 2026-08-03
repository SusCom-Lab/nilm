"""Official WindowGRU PyTorch plugin.

Official repository: https://github.com/nilmtk/nilmtk-contrib
Official revision: 14efd545e09d836e02159b23444350af92c6ee70
Official file: nilmtk_contrib/torch/WindowGRU.py
Local adaptations: canonical partitions, public validation MSE, complete raw-watt
timeline output, and repository checkpoint metadata.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from model_pipeline.api import (
    FitResult,
    InferenceContext,
    PredictionOutput,
    TrainingContext,
)
from model_pipeline.model_registry import register_model
from model_pipeline.models.seq2point.rnn import RNNBaseline


class FastReLUGRU(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        batch_first: bool = True,
        bidirectional: bool = False,
        return_sequences: bool = True,
    ) -> None:
        super().__init__()
        self.return_sequences = return_sequences
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=batch_first,
            bidirectional=bidirectional,
        )
        output_size = hidden_size * 2 if bidirectional else hidden_size
        self.activation_transform = nn.Sequential(
            nn.Linear(output_size, output_size),
            nn.ReLU(),
            nn.Linear(output_size, output_size),
        )

    def forward(self, inputs, hidden=None):
        if self.return_sequences:
            output, final_hidden = self.gru(inputs, hidden)
            shape = output.shape
            output = self.activation_transform(output.reshape(-1, shape[-1]))
            return output.reshape(shape), final_hidden
        _, final_hidden = self.gru(inputs, hidden)
        if final_hidden.dim() == 3:
            if final_hidden.size(0) == 2:
                final_hidden = torch.cat((final_hidden[0], final_hidden[1]), dim=1)
            else:
                final_hidden = final_hidden.squeeze(0)
        return None, self.activation_transform(final_hidden)


class WindowGRUDataset(Dataset):
    """Official forward-looking windows with end padding and scalar targets."""

    def __init__(
        self,
        partition,
        window_size: int,
        max_val: float,
        *,
        include_status: bool = False,
        domain_mapping: dict[str, int] | None = None,
    ) -> None:
        self.inputs = []
        self.targets = []
        self.domains = [] if domain_mapping is not None else None
        for series in partition.series:
            if len(series.appliances) != 1:
                raise ValueError("WindowGRU is a single-appliance model.")
            segments = series.segment_ids
            if segments is None:
                segments = np.zeros(len(series.aggregate), dtype=np.int64)
            start = 0
            while start < len(series.aggregate):
                end = start + 1
                while end < len(series.aggregate) and segments[end] == segments[start]:
                    end += 1
                aggregate = np.pad(
                    series.aggregate[start:end], (0, window_size - 1)
                )
                for index in range(end - start):
                    self.inputs.append(
                        torch.as_tensor(
                            aggregate[index : index + window_size] / max_val,
                            dtype=torch.float32,
                        )
                    )
                    target = series.appliance_power[start + index, 0] / max_val
                    if include_status:
                        if series.status is None:
                            raise ValueError("This WindowGRU variant requires public status labels.")
                        self.targets.append(
                            torch.tensor(
                                (target, series.status[start + index, 0]),
                                dtype=torch.float32,
                            )
                        )
                    else:
                        self.targets.append(torch.tensor(target, dtype=torch.float32))
                    if self.domains is not None:
                        self.domains.append(
                            torch.tensor(domain_mapping[series.household_id], dtype=torch.long)
                        )
                start = end

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        if self.domains is not None:
            return self.inputs[index], self.targets[index], self.domains[index]
        return self.inputs[index], self.targets[index]


@register_model("window_gru", aliases=("WindowGRU",), display_name="WindowGRU")
class WindowGRU(RNNBaseline):
    display_name = "WindowGRU"
    model_family = "rnn"
    target_type = "point"
    default_window_size = 99
    default_num_epochs = 10
    official_batch_size = 512

    def __init__(self, *, window_size: int = 99, max_val: float = 800.0):
        nn.Module.__init__(self)
        if window_size < 2 or max_val <= 0:
            raise ValueError("WindowGRU window_size and max_val must be positive.")
        self.window_size = int(window_size)
        self.output_size = 1
        self.output_offset = 0
        self.max_val = float(max_val)
        self._config = {"window_size": self.window_size, "max_val": self.max_val}
        self.normalization = None
        self.conv1 = nn.Conv1d(1, 16, kernel_size=4, padding=2, stride=1)
        self.gru1 = FastReLUGRU(
            16, 64, batch_first=True, bidirectional=True, return_sequences=True
        )
        self.dropout1 = nn.Dropout(0.5)
        self.gru2 = FastReLUGRU(
            128, 128, batch_first=True, bidirectional=True, return_sequences=False
        )
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256, 128)
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 1)
        self._initialize_weights()

    def _initialize_weights(self):
        for name, parameter in self.named_parameters():
            if "weight_ih" in name or "weight_hh" in name:
                nn.init.xavier_uniform_(parameter)
            elif "bias_ih" in name or "bias_hh" in name:
                nn.init.zeros_(parameter)
            elif "activation_transform" in name and "weight" in name:
                nn.init.xavier_uniform_(parameter)
            elif "activation_transform" in name and "bias" in name:
                nn.init.zeros_(parameter)
            elif "weight" in name and "conv1" in name:
                nn.init.xavier_uniform_(parameter)
            elif "bias" in name and "conv1" in name:
                nn.init.zeros_(parameter)
            elif "fc" in name and "weight" in name:
                nn.init.xavier_uniform_(parameter)
            elif "fc" in name and "bias" in name:
                nn.init.zeros_(parameter)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim == 2:
            inputs = inputs.unsqueeze(1)
        encoded = torch.relu(self.conv1(inputs)).permute(0, 2, 1)
        encoded, _ = self.gru1(encoded)
        encoded = self.dropout1(encoded)
        _, hidden = self.gru2(encoded)
        hidden = self.dropout2(hidden)
        hidden = self.dropout3(torch.relu(self.fc1(hidden)))
        return self.fc2(hidden).reshape(-1)

    @staticmethod
    def prepare_targets(targets):
        if isinstance(targets, torch.Tensor):
            return targets.float().reshape(-1)
        return np.asarray(targets, dtype=np.float32).reshape(-1)

    @staticmethod
    def prepare_outputs(outputs):
        if isinstance(outputs, torch.Tensor):
            return outputs.float().reshape(-1)
        return np.asarray(outputs, dtype=np.float32).reshape(-1)

    def fit(self, train_data, validation_data, context: TrainingContext) -> FitResult:
        del validation_data
        self.normalization = {
            "max_val": self.max_val,
            "appliances": train_data.series[0].appliances,
        }
        requires_domains = bool(getattr(self, "requires_domain_targets", False))
        domain_mapping = (
            {household: index for index, household in enumerate(sorted(train_data.household_ids))}
            if requires_domains else None
        )
        if requires_domains and len(domain_mapping) > int(self.num_domains):
            raise ValueError(
                f"{self.display_name} num_domains={self.num_domains} cannot represent "
                f"{len(domain_mapping)} training households."
            )
        loader = DataLoader(
            WindowGRUDataset(
                train_data,
                self.window_size,
                self.max_val,
                include_status=bool(getattr(self, "requires_status_targets", False)),
                domain_mapping=domain_mapping,
            ),
            batch_size=self.official_batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(context.seed),
        )
        self.to(context.device)
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=0.001,
            betas=(0.9, 0.999),
            eps=1e-7,
            weight_decay=0.0,
        )
        criterion = nn.MSELoss()
        history = []
        best_epoch = None
        best_mse = None
        checkpoint_path = None
        for epoch in range(1, context.num_epochs + 1):
            self.train()
            total = 0.0
            for batch in loader:
                inputs, targets = batch[:2]
                house_ids = batch[2] if requires_domains else None
                optimizer.zero_grad()
                inputs = inputs.to(context.device)
                targets = targets.to(context.device)
                loss_hook = getattr(self, "compute_loss", None)
                if callable(loss_hook):
                    loss_kwargs = {"criterion": criterion}
                    if house_ids is not None:
                        loss_kwargs["house_ids"] = house_ids.to(context.device)
                    loss_result = loss_hook(inputs, targets, **loss_kwargs)
                    loss = loss_result[0] if isinstance(loss_result, tuple) else loss_result
                else:
                    loss = criterion(self(inputs), targets)
                loss.backward()
                optimizer.step()
                total += float(loss.item())
            validation = context.validate_candidate(epoch=epoch, model=self)
            history.append(
                {
                    "epoch": epoch,
                    "training_loss": total / max(1, len(loader)),
                    "validation_mse": validation.mse,
                }
            )
            if validation.improved:
                best_epoch = epoch
                best_mse = validation.mse
                checkpoint_path = validation.checkpoint_path
            if validation.should_stop:
                break
        return FitResult(history, best_epoch, best_mse, checkpoint_path)

    def predict(self, inference_data, context: InferenceContext) -> PredictionOutput:
        if self.normalization is None:
            raise RuntimeError("WindowGRU must be fitted or loaded before prediction.")
        self.to(context.device)
        self.eval()
        timestamps, households, predictions = [], [], []
        with torch.inference_mode():
            for series in inference_data.series:
                segments = series.segment_ids
                if segments is None:
                    segments = np.zeros(len(series.aggregate), dtype=np.int64)
                output = np.empty(len(series.aggregate), dtype=np.float32)
                start = 0
                while start < len(series.aggregate):
                    end = start + 1
                    while end < len(series.aggregate) and segments[end] == segments[start]:
                        end += 1
                    aggregate = np.pad(
                        series.aggregate[start:end], (0, self.window_size - 1)
                    )
                    windows = np.stack(
                        [
                            aggregate[index : index + self.window_size]
                            for index in range(end - start)
                        ]
                    ) / self.max_val
                    batches = []
                    for offset in range(0, len(windows), context.batch_size):
                        inputs = torch.as_tensor(
                            windows[offset : offset + context.batch_size],
                            dtype=torch.float32,
                            device=context.device,
                        )
                        batches.append(self(inputs).cpu().numpy())
                    output[start:end] = np.concatenate(batches) * self.max_val
                    start = end
                timestamps.append(series.timestamps)
                households.append(np.repeat(series.household_id, len(series.timestamps)))
                predictions.append(output[:, None])
        return PredictionOutput(
            np.concatenate(timestamps),
            np.concatenate(predictions),
            tuple(self.normalization["appliances"]),
            np.concatenate(households),
        )


__all__ = ["FastReLUGRU", "WindowGRU", "WindowGRUDataset"]
