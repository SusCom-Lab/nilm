"""Official Seq2Point PyTorch plugin.

Official repository: https://github.com/nilmtk/nilmtk-contrib
Official revision: 14efd545e09d836e02159b23444350af92c6ee70
Official file: nilmtk_contrib/torch/seq2point.py
Local adaptations: canonical raw partitions, public validation-MSE callback,
unified full-timeline PredictionOutput, and repository checkpoint metadata.
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


class Seq2PointDataset(Dataset):
    """Official zero-padded, one-window-per-timestamp center-label dataset."""

    def __init__(
        self,
        partition: SupervisedPartition,
        *,
        window_size: int,
        mains_mean: float,
        mains_std: float,
        appliance_mean: float,
        appliance_std: float,
        include_status: bool = False,
        domain_mapping: dict[str, int] | None = None,
    ) -> None:
        if len(partition.series[0].appliances) != 1:
            raise ValueError("Seq2Point is a single-appliance model.")
        self.inputs: list[torch.Tensor] = []
        self.targets: list[torch.Tensor] = []
        self.domains: list[torch.Tensor] | None = [] if domain_mapping is not None else None
        half = window_size // 2
        for series in partition.series:
            segment_ids = (
                series.segment_ids
                if series.segment_ids is not None
                else np.zeros(len(series.aggregate), dtype=np.int64)
            )
            start = 0
            while start < len(series.aggregate):
                end = start + 1
                while end < len(series.aggregate) and segment_ids[end] == segment_ids[start]:
                    end += 1
                aggregate = np.pad(
                    series.aggregate[start:end],
                    (half, half),
                    mode="constant",
                )
                for index in range(end - start):
                    window = aggregate[index : index + window_size]
                    self.inputs.append(
                        torch.as_tensor(
                            (window - mains_mean) / mains_std,
                            dtype=torch.float32,
                        )
                    )
                    target = (
                        series.appliance_power[start + index, 0] - appliance_mean
                    ) / appliance_std
                    if include_status:
                        if series.status is None:
                            raise ValueError(
                                "This model requires a status column in every training CSV."
                            )
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

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int):
        if self.domains is not None:
            return self.inputs[index], self.targets[index], self.domains[index]
        return self.inputs[index], self.targets[index]


@register_model(
    "seq2point",
    aliases=("seq2point_simple", "Seq2Point"),
    display_name="Seq2Point",
)
class Seq2Point(nn.Module):
    display_name = "Seq2Point"
    model_family = "seq2point"
    target_type = "point"
    supports_gradient = True
    is_joint_classical = False
    default_window_size = 99
    default_num_epochs = 10
    official_batch_size = 512
    official_learning_rate = 0.001
    official_mains_mean = 1800.0
    official_mains_std = 600.0

    def __init__(self, *, window_size: int = 99, hidden_dim: int = 1024) -> None:
        super().__init__()
        if window_size % 2 == 0:
            raise ValueError("Seq2Point window_size must be odd.")
        self.window_size = window_size
        self.output_size = 1
        self.output_offset = window_size // 2
        self._config = {"window_size": window_size, "hidden_dim": hidden_dim}
        self.normalization: dict[str, Any] | None = None
        conv_reduction = (10 - 1) + (8 - 1) + (6 - 1) + (5 - 1) + (5 - 1)
        if window_size <= conv_reduction:
            raise ValueError(f"Seq2Point window_size must be greater than {conv_reduction}.")
        flattened = 50 * (window_size - conv_reduction)
        self.network = nn.Sequential(
            nn.Conv1d(1, 30, kernel_size=10),
            nn.ReLU(),
            nn.Conv1d(30, 30, kernel_size=8),
            nn.ReLU(),
            nn.Conv1d(30, 40, kernel_size=6),
            nn.ReLU(),
            nn.Conv1d(40, 50, kernel_size=5),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(50, 50, kernel_size=5),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Flatten(),
            nn.Linear(flattened, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim == 2:
            inputs = inputs.unsqueeze(1)
        return self.network(inputs).reshape(-1)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network[:-1](inputs)

    def decode(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.network[-1](hidden)

    def prepare_targets(self, targets):
        targets = targets.float() if isinstance(targets, torch.Tensor) else np.asarray(targets, dtype=np.float32)
        return targets.reshape(-1)

    def prepare_outputs(self, outputs):
        outputs = outputs.float() if isinstance(outputs, torch.Tensor) else np.asarray(outputs, dtype=np.float32)
        return outputs.reshape(-1)

    def _set_training_normalization(self, train_data: SupervisedPartition) -> None:
        targets = np.concatenate(
            [series.appliance_power[:, 0] for series in train_data.series]
        )
        appliance_std = float(np.std(targets))
        if not np.isfinite(appliance_std) or appliance_std < 1.0:
            appliance_std = 100.0
        self.normalization = {
            "mains_mean": self.official_mains_mean,
            "mains_std": self.official_mains_std,
            "appliance_mean": float(np.mean(targets)),
            "appliance_std": appliance_std,
            "appliances": train_data.series[0].appliances,
        }

    def fit(
        self,
        train_data: SupervisedPartition,
        validation_data: InferencePartition,
        context: TrainingContext,
    ) -> FitResult:
        del validation_data
        self._set_training_normalization(train_data)
        stats = self.normalization
        configure_normalization = getattr(self, "set_normalisation_stats", None)
        if callable(configure_normalization):
            configure_normalization(
                {
                    "appliance_mean": stats["appliance_mean"],
                    "appliance_std": stats["appliance_std"],
                }
            )
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
        dataset = Seq2PointDataset(
            train_data,
            window_size=self.window_size,
            mains_mean=stats["mains_mean"],
            mains_std=stats["mains_std"],
            appliance_mean=stats["appliance_mean"],
            appliance_std=stats["appliance_std"],
            include_status=bool(getattr(self, "requires_status_targets", False)),
            domain_mapping=domain_mapping,
        )
        generator = torch.Generator().manual_seed(context.seed)
        loader = DataLoader(
            dataset,
            batch_size=self.official_batch_size,
            shuffle=True,
            generator=generator,
        )
        self.to(context.device)
        optimizer = torch.optim.Adam(
            self.parameters(), lr=float(self.official_learning_rate)
        )
        criterion = nn.MSELoss()
        history: list[dict[str, Any]] = []
        best_epoch = None
        best_mse = None
        checkpoint_path = None

        for epoch in range(1, context.num_epochs + 1):
            self.train()
            total_loss = 0.0
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
                    loss_output = loss_hook(inputs, targets, **loss_kwargs)
                    loss = loss_output[0] if isinstance(loss_output, tuple) else loss_output
                else:
                    loss = criterion(self(inputs), self.prepare_targets(targets))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += float(loss.item())
            training_loss = total_loss / max(1, len(loader))
            validation = context.validate_candidate(epoch=epoch, model=self)
            history.append(
                {
                    "epoch": epoch,
                    "training_loss": training_loss,
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

    def predict(
        self,
        inference_data: InferencePartition,
        context: InferenceContext,
    ) -> PredictionOutput:
        if self.normalization is None:
            raise RuntimeError("Seq2Point must be fitted or loaded before prediction.")
        stats = self.normalization
        self.to(context.device)
        self.eval()
        timestamp_parts = []
        household_parts = []
        power_parts = []
        half = self.window_size // 2
        with torch.inference_mode():
            for series in inference_data.series:
                if len(series.appliances) != 1:
                    raise ValueError("Seq2Point expects exactly one target appliance.")
                segment_ids = (
                    series.segment_ids
                    if series.segment_ids is not None
                    else np.zeros(len(series.aggregate), dtype=np.int64)
                )
                series_prediction = np.empty(len(series.aggregate), dtype=np.float32)
                start = 0
                while start < len(series.aggregate):
                    end = start + 1
                    while end < len(series.aggregate) and segment_ids[end] == segment_ids[start]:
                        end += 1
                    aggregate = np.pad(
                        series.aggregate[start:end],
                        (half, half),
                        mode="constant",
                    )
                    windows = np.stack(
                        [aggregate[index : index + self.window_size] for index in range(end - start)]
                    )
                    windows = (windows - stats["mains_mean"]) / stats["mains_std"]
                    batches = []
                    for offset in range(0, len(windows), context.batch_size):
                        inputs = torch.as_tensor(
                            windows[offset : offset + context.batch_size],
                            dtype=torch.float32,
                            device=context.device,
                        )
                        batches.append(self.prepare_outputs(self(inputs)).cpu().numpy())
                    normalized = np.concatenate(batches)
                    series_prediction[start:end] = (
                        normalized * stats["appliance_std"] + stats["appliance_mean"]
                    )
                    start = end
                timestamp_parts.append(series.timestamps)
                household_parts.append(np.repeat(series.household_id, len(series.timestamps)))
                power_parts.append(series_prediction[:, None])
        return PredictionOutput(
            timestamps=np.concatenate(timestamp_parts),
            power=np.concatenate(power_parts, axis=0),
            appliances=tuple(stats["appliances"]),
            household_ids=np.concatenate(household_parts),
        )

    def save(self, path: str, *, metadata: dict[str, Any] | None = None) -> None:
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

    def load(self, path: str, device: str) -> None:
        try:
            checkpoint = torch.load(path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location=device)
        self.load_state_dict(checkpoint["model_state"])
        self.normalization = checkpoint["normalization"]
        self.to(device)

    def get_window_size(self) -> int:
        return self.window_size

    def getWindowSize(self) -> int:
        return self.window_size

    def get_output_size(self) -> int:
        return self.output_size

    def get_output_offset(self) -> int:
        return self.output_offset

    def get_target_type(self) -> str:
        return self.target_type

    def get_init_kwargs(self) -> dict[str, Any]:
        return dict(self._config)

    def freeze_for_finetuning(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False
        for module in self.modules():
            if isinstance(module, nn.Linear):
                for parameter in module.parameters():
                    parameter.requires_grad = True
