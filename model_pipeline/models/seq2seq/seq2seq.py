"""Official Seq2Seq CNN PyTorch plugin.

Official repository: https://github.com/nilmtk/nilmtk-contrib
Official revision: 14efd545e09d836e02159b23444350af92c6ee70
Official file: nilmtk_contrib/torch/seq2seq.py
Local adaptations: canonical partitions, public validation MSE, full-timeline
raw-watt predictions, and repository checkpoint metadata.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from model_pipeline.api import (
    FitResult,
    InferenceContext,
    InferencePartition,
    PredictionOutput,
    SupervisedPartition,
    TrainingContext,
)
from model_pipeline.model_registry import register_model


class Seq2SeqDataset(Dataset):
    """Official centered zero-padded sequence-window dataset."""

    def __init__(self, partition, window_size, stats) -> None:
        if len(partition.series[0].appliances) != 1:
            raise ValueError("Seq2Seq is a single-appliance model.")
        self.inputs = []
        self.targets = []
        half = window_size // 2
        for series in partition.series:
            segment_ids = series.segment_ids
            if segment_ids is None:
                segment_ids = np.zeros(len(series.aggregate), dtype=np.int64)
            start = 0
            while start < len(series.aggregate):
                end = start + 1
                while end < len(series.aggregate) and segment_ids[end] == segment_ids[start]:
                    end += 1
                mains = np.pad(series.aggregate[start:end], (half, half))
                target = np.pad(series.appliance_power[start:end, 0], (half, half))
                for index in range(end - start):
                    self.inputs.append(
                        torch.as_tensor(
                            (mains[index : index + window_size] - stats["mains_mean"])
                            / stats["mains_std"],
                            dtype=torch.float32,
                        )
                    )
                    self.targets.append(
                        torch.as_tensor(
                            (target[index : index + window_size] - stats["appliance_mean"])
                            / stats["appliance_std"],
                            dtype=torch.float32,
                        )
                    )
                start = end

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]


@register_model("seq2seq", aliases=("Seq2Seq",), display_name="Seq2Seq")
class Seq2SeqCNN(nn.Module):
    display_name = "Seq2Seq"
    model_family = "seq2seq"
    target_type = "sequence"
    supports_gradient = True
    is_joint_classical = False
    default_window_size = 99
    default_num_epochs = 10
    official_batch_size = 512
    official_mains_mean = 1800.0
    official_mains_std = 600.0

    def __init__(self, *, window_size: int = 99, hidden_dim: int = 1024) -> None:
        super().__init__()
        if window_size != 99 or hidden_dim != 1024:
            raise ValueError("Seq2Seq uses official window_size=99 and hidden_dim=1024.")
        self.window_size = window_size
        self.output_size = window_size
        self.output_offset = 0
        self.hidden_dim = hidden_dim
        self._config = {"window_size": window_size, "hidden_dim": hidden_dim}
        self.normalization: dict[str, Any] | None = None
        length = (window_size - 10) // 2 + 1
        length = (length - 8) // 2 + 1
        length = length - 6 + 1
        length = length - 5 + 1
        length = length - 5 + 1
        self.conv1 = nn.Conv1d(1, 30, 10, stride=2)
        self.conv2 = nn.Conv1d(30, 30, 8, stride=2)
        self.conv3 = nn.Conv1d(30, 40, 6)
        self.conv4 = nn.Conv1d(40, 50, 5)
        self.dropout1 = nn.Dropout(0.2)
        self.conv5 = nn.Conv1d(50, 50, 5)
        self.dropout2 = nn.Dropout(0.2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(50 * length, hidden_dim)
        self.dropout3 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_dim, window_size)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def encode(self, inputs):
        inputs = inputs.unsqueeze(1)
        inputs = torch.relu(self.conv1(inputs))
        inputs = torch.relu(self.conv2(inputs))
        inputs = torch.relu(self.conv3(inputs))
        inputs = torch.relu(self.conv4(inputs))
        inputs = self.dropout1(inputs)
        inputs = torch.relu(self.conv5(inputs))
        inputs = self.dropout2(inputs)
        inputs = self.flatten(inputs)
        return self.dropout3(torch.relu(self.fc1(inputs)))

    def forward(self, inputs):
        return self.fc2(self.encode(inputs))

    def _set_normalization(self, train_data):
        targets = np.concatenate([item.appliance_power[:, 0] for item in train_data.series])
        std = float(np.std(targets))
        if not np.isfinite(std) or std < 1.0:
            std = 100.0
        self.normalization = {
            "mains_mean": self.official_mains_mean,
            "mains_std": self.official_mains_std,
            "appliance_mean": float(np.mean(targets)),
            "appliance_std": std,
            "appliances": train_data.series[0].appliances,
        }

    def fit(self, train_data, validation_data, context: TrainingContext) -> FitResult:
        del validation_data
        self._set_normalization(train_data)
        dataset = Seq2SeqDataset(train_data, self.window_size, self.normalization)
        loader = DataLoader(
            dataset,
            batch_size=self.official_batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(context.seed),
        )
        self.to(context.device)
        optimizer = torch.optim.Adam(
            self.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-7
        )
        criterion = nn.MSELoss()
        history = []
        best_epoch = None
        best_mse = None
        checkpoint_path = None
        for epoch in range(1, context.num_epochs + 1):
            self.train()
            total = 0.0
            for inputs, targets in loader:
                optimizer.zero_grad()
                loss = criterion(
                    self(inputs.to(context.device)), targets.to(context.device)
                )
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

    def _predict_segment(self, aggregate, context):
        stats = self.normalization
        half = self.window_size // 2
        padded = np.pad(aggregate, (half, half))
        windows = np.stack(
            [padded[index : index + self.window_size] for index in range(len(aggregate))]
        )
        windows = (windows - stats["mains_mean"]) / stats["mains_std"]
        predicted = []
        with torch.inference_mode():
            for offset in range(0, len(windows), context.batch_size):
                inputs = torch.as_tensor(
                    windows[offset : offset + context.batch_size],
                    dtype=torch.float32,
                    device=context.device,
                )
                predicted.append(self(inputs).cpu().numpy())
        predicted = np.concatenate(predicted)
        predicted = predicted * stats["appliance_std"] + stats["appliance_mean"]
        sums = np.zeros(len(padded), dtype=np.float64)
        counts = np.zeros(len(padded), dtype=np.float64)
        for index, window in enumerate(predicted):
            sums[index : index + self.window_size] += window
            counts[index : index + self.window_size] += 1
        return (sums[half : half + len(aggregate)] / counts[half : half + len(aggregate)]).astype(
            np.float32
        )

    def predict(self, inference_data: InferencePartition, context: InferenceContext):
        if self.normalization is None:
            raise RuntimeError("Seq2Seq must be fitted or loaded before prediction.")
        self.to(context.device)
        self.eval()
        timestamps = []
        households = []
        predictions = []
        for series in inference_data.series:
            segment_ids = series.segment_ids
            if segment_ids is None:
                segment_ids = np.zeros(len(series.aggregate), dtype=np.int64)
            output = np.empty(len(series.aggregate), dtype=np.float32)
            start = 0
            while start < len(series.aggregate):
                end = start + 1
                while end < len(series.aggregate) and segment_ids[end] == segment_ids[start]:
                    end += 1
                output[start:end] = self._predict_segment(series.aggregate[start:end], context)
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

    def save(self, path, *, metadata=None):
        torch.save(
            {
                "model_key": self._registry_key,
                "model_name": self.display_name,
                "init_kwargs": dict(self._config),
                "model_state": self.state_dict(),
                "normalization": self.normalization,
                "metadata": dict(metadata or {}),
            },
            path,
        )

    def load(self, path, device):
        try:
            checkpoint = torch.load(path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location=device)
        self.load_state_dict(checkpoint["model_state"])
        self.normalization = checkpoint["normalization"]
        self.to(device)

    def get_window_size(self): return self.window_size
    def getWindowSize(self): return self.window_size
    def get_output_size(self): return self.output_size
    def get_output_offset(self): return self.output_offset
    def get_target_type(self): return self.target_type
    def get_init_kwargs(self): return dict(self._config)

    def freeze_for_finetuning(self):
        for parameter in self.parameters():
            parameter.requires_grad = False
        for module in self.modules():
            if isinstance(module, nn.Linear):
                for parameter in module.parameters():
                    parameter.requires_grad = True
