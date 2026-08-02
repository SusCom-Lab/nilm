"""
State-aware Seq2Point CNN variant for NILM.

Base source: https://github.com/nilmtk/nilmtk-contrib/blob/14efd545e09d836e02159b23444350af92c6ee70/nilmtk_contrib/torch/seq2point.py
Implementation status: local state-supervised research adaptation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_pipeline.model_registry import register_model
from model_pipeline.models.seq2point.seq2point import Seq2Point


def power_to_status(targets: torch.Tensor, threshold: float) -> torch.Tensor:
    return (targets.reshape(-1) >= float(threshold)).to(dtype=torch.float32)


@register_model(
    "state_aware_seq2point",
    aliases=("sa_seq2point", "State-aware Seq2Point", "StateAwareSeq2Point"),
    display_name="StateAwareSeq2Point",
)
class StateAwareSeq2Point(Seq2Point):
    display_name = "StateAwareSeq2Point"
    model_family = "seq2point"
    target_type = "point"
    requires_status_targets = True
    default_window_size = 599
    default_num_epochs = 10

    def __init__(
        self,
        *,
        window_size: int = 599,
        hidden_dim: int = 1024,
        state_threshold: float = 0.0,
        state_loss_weight: float = 0.1,
        state_pos_weight: float | None = None,
    ):
        nn.Module.__init__(self)
        self.window_size = window_size
        self.output_size = 1
        self.output_offset = window_size // 2
        self._config = {
            "window_size": window_size,
            "hidden_dim": hidden_dim,
            "state_threshold": state_threshold,
            "state_loss_weight": state_loss_weight,
            "state_pos_weight": state_pos_weight,
        }
        self.normalization = None
        self.state_threshold = float(state_threshold)
        self.state_loss_weight = float(state_loss_weight)
        self.state_pos_weight = None if state_pos_weight is None else float(state_pos_weight)

        conv_reduction = (10 - 1) + (8 - 1) + (6 - 1) + (5 - 1) + (5 - 1)
        flattened = 50 * (self.window_size - conv_reduction)
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 30, kernel_size=10, stride=1),
            nn.ReLU(),
            nn.Conv1d(30, 30, kernel_size=8, stride=1),
            nn.ReLU(),
            nn.Conv1d(30, 40, kernel_size=6, stride=1),
            nn.ReLU(),
            nn.Conv1d(40, 50, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(50, 50, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Flatten(),
        )
        self.representation = nn.Sequential(
            nn.Linear(flattened, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.power_head = nn.Linear(hidden_dim, 1)
        self.state_head = nn.Linear(hidden_dim, 1)
        self.last_state_logits: torch.Tensor | None = None

    def prepare_targets(self, targets):
        if isinstance(targets, torch.Tensor) and self.target_type == "point" and targets.ndim == 2 and targets.size(-1) == 2:
            return targets.float()
        return super().prepare_targets(targets)

    def _split_targets(self, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        if targets.ndim == 2 and targets.size(-1) == 2:
            return targets[:, 0].reshape(-1), targets[:, 1].reshape(-1).float()
        return super().prepare_targets(targets), None

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        return self.representation(self.encoder(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.power_head(z)

    def compute_loss(self, inputs: torch.Tensor, targets: torch.Tensor, criterion) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(inputs)
        outputs = self.power_head(z)
        prepared_outputs = self.prepare_outputs(outputs)
        prepared_targets, provided_state_targets = self._split_targets(targets)

        power_loss = criterion(prepared_outputs, prepared_targets)
        if provided_state_targets is None:
            return power_loss, outputs

        state_targets = provided_state_targets.to(device=inputs.device)
        state_logits = self.state_head(z).reshape(-1)
        self.last_state_logits = state_logits

        pos_weight = None
        if self.state_pos_weight is not None:
            pos_weight = torch.as_tensor(self.state_pos_weight, dtype=state_logits.dtype, device=state_logits.device)
        state_loss = F.binary_cross_entropy_with_logits(state_logits, state_targets, pos_weight=pos_weight)
        return power_loss + (self.state_loss_weight * state_loss), outputs

    def compute_selection_loss(self, inputs: torch.Tensor, targets: torch.Tensor, criterion) -> torch.Tensor:
        outputs = self.prepare_outputs(self(inputs))
        prepared_targets, _ = self._split_targets(targets)
        return criterion(outputs, prepared_targets)
