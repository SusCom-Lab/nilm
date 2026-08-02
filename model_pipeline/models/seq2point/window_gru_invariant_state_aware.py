"""
House-invariant state-aware WindowGRU model.

Base source: https://github.com/nilmtk/nilmtk-contrib/blob/14efd545e09d836e02159b23444350af92c6ee70/nilmtk_contrib/torch/WindowGRU.py
Implementation status: local domain-invariant research adaptation.

This keeps the original WindowGRU encoder path and adds a state head plus an
optional adversarial house head. With house_loss_weight=0 it reduces to a
state-aware WindowGRU.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_pipeline.model_registry import register_model
from model_pipeline.models.base_model import TorchNILMModel
from model_pipeline.models.seq2point.seq2point_invariant_state_aware import GradientReversal


@register_model(
    "invariant_state_aware_window_gru",
    aliases=("InvariantStateAwareWindowGRU", "isa_window_gru"),
    display_name="InvariantStateAwareWindowGRU",
)
class InvariantStateAwareWindowGRU(TorchNILMModel):
    display_name = "InvariantStateAwareWindowGRU"
    model_family = "rnn"
    target_type = "point"

    def __init__(
        self,
        *,
        window_size: int = 599,
        hidden_dim: int = 128,
        state_threshold: float = 0.0,
        state_loss_weight: float = 0.05,
        state_pos_weight: float | None = None,
        num_domains: int = 2,
        house_loss_weight: float = 0.003,
        grl_lambda: float = 1.0,
        **kwargs,
    ):
        super().__init__(
            window_size=window_size,
            hidden_dim=hidden_dim,
            state_threshold=state_threshold,
            state_loss_weight=state_loss_weight,
            state_pos_weight=state_pos_weight,
            num_domains=num_domains,
            house_loss_weight=house_loss_weight,
            grl_lambda=grl_lambda,
            **kwargs,
        )
        self.state_threshold = float(state_threshold)
        self.state_loss_weight = float(state_loss_weight)
        self.state_pos_weight = None if state_pos_weight is None else float(state_pos_weight)
        self.num_domains = int(num_domains)
        self.house_loss_weight = float(house_loss_weight)
        self.grl_lambda = float(grl_lambda)

        self.conv1 = nn.Conv1d(1, 16, kernel_size=4, padding=2)
        self.gru1 = nn.GRU(16, 64, batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(0.5)
        self.gru2 = nn.GRU(128, 128, batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256, hidden_dim)
        self.dropout3 = nn.Dropout(0.5)
        self.power_head = nn.Linear(hidden_dim, 1)
        self.state_head = nn.Linear(hidden_dim, 1)
        self.house_head = nn.Linear(hidden_dim, self.num_domains)
        self.last_state_logits: torch.Tensor | None = None

    def prepare_targets(self, targets):
        if isinstance(targets, torch.Tensor) and targets.ndim == 2 and targets.size(-1) == 2:
            return targets.float()
        return super().prepare_targets(targets)

    def _split_targets(self, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        if targets.ndim == 2 and targets.size(-1) == 2:
            return targets[:, 0].reshape(-1), targets[:, 1].reshape(-1).float()
        return super().prepare_targets(targets), None

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = torch.relu(self.conv1(x))
        x = x.permute(0, 2, 1)
        x, _ = self.gru1(x)
        x = self.dropout1(x)
        _, hidden = self.gru2(x)
        x = torch.cat([hidden[-2], hidden[-1]], dim=1)
        x = self.dropout2(x)
        x = torch.relu(self.fc1(x))
        return self.dropout3(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.power_head(self.encode(x))

    def _state_loss(self, state_logits: torch.Tensor, state_targets: torch.Tensor) -> torch.Tensor:
        pos_weight = None
        if self.state_pos_weight is not None:
            pos_weight = torch.as_tensor(self.state_pos_weight, dtype=state_logits.dtype, device=state_logits.device)
        return F.binary_cross_entropy_with_logits(state_logits, state_targets, pos_weight=pos_weight)

    def compute_loss(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        house_ids: torch.Tensor | None = None,
        criterion=None,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if criterion is None:
            criterion = nn.MSELoss()

        z = self.encode(inputs)
        outputs = self.power_head(z)
        prepared_outputs = self.prepare_outputs(outputs)
        prepared_targets, state_targets = self._split_targets(targets)
        power_loss = criterion(prepared_outputs, prepared_targets)

        zero = torch.zeros((), dtype=power_loss.dtype, device=power_loss.device)
        if state_targets is None:
            if house_ids is None:
                return power_loss, outputs
            return power_loss, outputs, {
                "power_loss": power_loss.detach(),
                "state_loss": zero.detach(),
                "house_loss": zero.detach(),
            }

        state_targets = state_targets.to(device=inputs.device)
        state_logits = self.state_head(z).reshape(-1)
        self.last_state_logits = state_logits
        state_loss = self._state_loss(state_logits, state_targets)

        if house_ids is None:
            house_loss = zero
        else:
            reversed_z = GradientReversal.apply(z, self.grl_lambda)
            house_logits = self.house_head(reversed_z)
            house_loss = F.cross_entropy(house_logits, house_ids.to(device=inputs.device).long())

        total_loss = power_loss + self.state_loss_weight * state_loss + self.house_loss_weight * house_loss
        if house_ids is None:
            return total_loss, outputs
        return total_loss, outputs, {
            "power_loss": power_loss.detach(),
            "state_loss": state_loss.detach(),
            "house_loss": house_loss.detach(),
        }

    def compute_selection_loss(self, inputs: torch.Tensor, targets: torch.Tensor, criterion) -> torch.Tensor:
        outputs = self.prepare_outputs(self(inputs))
        prepared_targets, _ = self._split_targets(targets)
        return criterion(outputs, prepared_targets)
