"""Subtask-gated sequence-to-point network for NILM.

Official source: https://github.com/nilmtk/nilmtk-contrib/blob/14efd545e09d836e02159b23444350af92c6ee70/nilmtk_contrib/torch/sgn.py
Local adaptation: repository normalization, registry, and plugin interfaces.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_pipeline.model_registry import register_model
from model_pipeline.models.seq2point.seq2point import Seq2Point


PAPER_KERNEL_SIZES = (10, 8, 6, 5, 5, 5)
PAPER_FILTERS = (30, 30, 40, 50, 50, 50)
PAPER_ON_POWER_THRESHOLD = 15.0


class SGNTower(nn.Module):
    """One paper-shaped CNN subnetwork with a scalar output head."""

    def __init__(self, window_size: int, hidden_dim: int = 1024, dropout: float = 0.0):
        super().__init__()
        reduced_length = int(window_size) - sum(size - 1 for size in PAPER_KERNEL_SIZES)
        if reduced_length < 1:
            minimum = 1 + sum(size - 1 for size in PAPER_KERNEL_SIZES)
            raise ValueError(f"window_size must be at least {minimum}.")

        channels = (1, *PAPER_FILTERS[:-1])
        self.convolutions = nn.ModuleList(
            nn.Conv1d(in_channels, out_channels, kernel_size)
            for in_channels, out_channels, kernel_size in zip(
                channels, PAPER_FILTERS, PAPER_KERNEL_SIZES, strict=True
            )
        )
        self.feature_dropout = nn.Dropout(dropout)
        self.hidden = nn.Linear(PAPER_FILTERS[-1] * reduced_length, hidden_dim)
        self.hidden_dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, 1)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for layer in (*self.convolutions, self.hidden):
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)
        nn.init.kaiming_normal_(self.output.weight, nonlinearity="linear")
        nn.init.zeros_(self.output.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoded = inputs
        for convolution in self.convolutions:
            encoded = torch.relu(convolution(encoded))
        encoded = self.feature_dropout(encoded).flatten(1)
        encoded = self.hidden_dropout(torch.relu(self.hidden(encoded)))
        return self.output(encoded)


@register_model(
    "sgn",
    aliases=("SGN", "Subtask Gated Network"),
    display_name="SGN",
)
class SGN(Seq2Point):
    """Independent regression and classification towers joined by a soft gate."""

    display_name = "SGN"
    model_family = "seq2point"
    target_type = "point"
    default_window_size = 299
    default_num_epochs = 10
    official_batch_size = 16
    official_learning_rate = 1e-4
    requires_status_targets = True

    def __init__(
        self,
        *,
        window_size: int = 299,
        hidden_dim: int = 1024,
        dropout: float = 0.0,
        classification_weight: float = 1.0,
        on_power_threshold: float = PAPER_ON_POWER_THRESHOLD,
    ):
        window_size = int(window_size)
        hidden_dim = int(hidden_dim)
        dropout = float(dropout)
        classification_weight = float(classification_weight)
        on_power_threshold = float(on_power_threshold)
        minimum = 1 + sum(size - 1 for size in PAPER_KERNEL_SIZES)
        if window_size % 2 == 0:
            raise ValueError("window_size must be odd.")
        if window_size < minimum:
            raise ValueError(f"window_size must be at least {minimum}.")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be a positive integer.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the interval [0, 1).")
        if classification_weight <= 0.0:
            raise ValueError("classification_weight must be greater than zero.")
        if on_power_threshold < 0.0:
            raise ValueError("on_power_threshold must be non-negative.")

        nn.Module.__init__(self)
        self.window_size = window_size
        self.output_size = 1
        self.output_offset = window_size // 2
        self._config = {
            "window_size": window_size,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "classification_weight": classification_weight,
            "on_power_threshold": on_power_threshold,
        }
        self.normalization = None
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.classification_weight = classification_weight
        self.on_power_threshold = on_power_threshold
        self.regression_tower = SGNTower(window_size, hidden_dim, dropout)
        self.classification_tower = SGNTower(window_size, hidden_dim, dropout)
        self.register_buffer("target_mean", torch.tensor(float("nan")))
        self.register_buffer("target_std", torch.tensor(float("nan")))

    def set_normalisation_stats(self, stats: dict[str, float]) -> None:
        mean = float(stats["appliance_mean"])
        std = float(stats["appliance_std"])
        if not math.isfinite(mean) or not math.isfinite(std) or std <= 0.0:
            raise ValueError("SGN requires finite appliance normalization with positive std.")
        self.target_mean.copy_(self.target_mean.new_tensor(mean))
        self.target_std.copy_(self.target_std.new_tensor(std))

    def _check_target_normalisation(self) -> None:
        if not torch.isfinite(self.target_mean) or not torch.isfinite(self.target_std):
            raise RuntimeError("SGN target normalization is not configured.")

    def training_outputs(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._check_target_normalisation()
        encoded_inputs = inputs.unsqueeze(1)
        regression = self.regression_tower(encoded_inputs)
        logits = self.classification_tower(encoded_inputs)
        probability = torch.sigmoid(logits)
        off_normalized = -self.target_mean / self.target_std
        gated = off_normalized + probability * (regression - off_normalized)
        return gated, regression, logits

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        gated, _, _ = self.training_outputs(inputs)
        return gated

    def compute_loss(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        criterion,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gated, _, logits = self.training_outputs(inputs)
        prepared_outputs = self.prepare_outputs(gated)
        if targets.ndim != 2 or targets.size(-1) != 2:
            raise ValueError("SGN requires public power and status targets.")
        prepared_targets = super().prepare_targets(targets[:, 0])
        on_targets = super().prepare_targets(targets[:, 1]).to(logits.dtype)
        output_loss = criterion(prepared_outputs, prepared_targets)
        classification_loss = F.binary_cross_entropy_with_logits(
            logits.reshape(-1), on_targets.reshape(-1)
        )
        return output_loss + self.classification_weight * classification_loss, gated

    def compute_selection_loss(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        criterion,
    ) -> torch.Tensor:
        outputs = self.prepare_outputs(self(inputs))
        power_targets = targets[:, 0] if targets.ndim == 2 else targets
        return criterion(outputs, super().prepare_targets(power_targets))


__all__ = [
    "PAPER_FILTERS",
    "PAPER_KERNEL_SIZES",
    "PAPER_ON_POWER_THRESHOLD",
    "SGN",
    "SGNTower",
]
