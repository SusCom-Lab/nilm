"""
State-conditioned invariant Seq2Point model for cross-house NILM.

Base source: https://github.com/nilmtk/nilmtk-contrib/blob/14efd545e09d836e02159b23444350af92c6ee70/nilmtk_contrib/torch/seq2point.py
Implementation status: local domain-invariant research adaptation.

The model separates state-related features from amplitude-related features:

    x -> CNN encoder -> shared h
                         |-> z_s -> state head
                         |-> z_a
                    [z_s, z_a] -> power head

The adversarial house/domain loss is applied only to z_s and is computed
within each on/off state group.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_pipeline.model_registry import register_model
from model_pipeline.models.seq2point.seq2point import Seq2Point
from model_pipeline.models.seq2point.seq2point_invariant_state_aware import GradientReversal


@register_model(
    "state_conditioned_invariant_seq2point",
    aliases=("StateConditionedInvariantSeq2Point", "sci_seq2point"),
    display_name="StateConditionedInvariantSeq2Point",
)
class StateConditionedInvariantSeq2Point(Seq2Point):
    display_name = "StateConditionedInvariantSeq2Point"
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
        state_dim: int | None = None,
        amplitude_dim: int | None = None,
        state_threshold: float = 0.0,
        state_loss_weight: float = 0.05,
        state_pos_weight: float | None = None,
        num_domains: int = 2,
        house_loss_weight: float = 0.003,
        grl_lambda: float = 1.0,
    ):
        if state_dim is None and amplitude_dim is None:
            state_dim = hidden_dim // 2
            amplitude_dim = hidden_dim - state_dim
        elif state_dim is None:
            state_dim = hidden_dim - int(amplitude_dim)
        elif amplitude_dim is None:
            amplitude_dim = hidden_dim - int(state_dim)
        state_dim = int(state_dim)
        amplitude_dim = int(amplitude_dim)
        if state_dim <= 0 or amplitude_dim <= 0:
            raise ValueError("state_dim and amplitude_dim must both be positive.")

        nn.Module.__init__(self)
        self.window_size = window_size
        self.output_size = 1
        self.output_offset = window_size // 2
        self._config = {
            "window_size": window_size, "hidden_dim": hidden_dim,
            "state_dim": state_dim, "amplitude_dim": amplitude_dim,
            "state_threshold": state_threshold, "state_loss_weight": state_loss_weight,
            "state_pos_weight": state_pos_weight, "num_domains": num_domains,
            "house_loss_weight": house_loss_weight, "grl_lambda": grl_lambda,
        }
        self.normalization = None
        self.state_dim = state_dim
        self.amplitude_dim = amplitude_dim
        self.state_threshold = float(state_threshold)
        self.state_loss_weight = float(state_loss_weight)
        self.state_pos_weight = None if state_pos_weight is None else float(state_pos_weight)
        self.num_domains = int(num_domains)
        self.house_loss_weight = float(house_loss_weight)
        self.grl_lambda = float(grl_lambda)

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
        self.shared_representation = nn.Sequential(
            nn.Linear(flattened, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(hidden_dim, state_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.amplitude_encoder = nn.Sequential(
            nn.Linear(hidden_dim, amplitude_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.state_head = nn.Linear(state_dim, 1)
        self.power_head = nn.Linear(state_dim + amplitude_dim, 1)
        self.house_head = nn.Linear(state_dim, self.num_domains)
        self.last_state_logits: torch.Tensor | None = None
        self._initialize_weights()

    def prepare_targets(self, targets):
        if isinstance(targets, torch.Tensor) and self.target_type == "point" and targets.ndim == 2 and targets.size(-1) == 2:
            return targets.float()
        return super().prepare_targets(targets)

    def _split_targets(self, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        if targets.ndim == 2 and targets.size(-1) == 2:
            return targets[:, 0].reshape(-1), targets[:, 1].reshape(-1).float()
        return super().prepare_targets(targets), None

    def encode_shared(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        return self.shared_representation(self.encoder(x))

    def encode_state_amplitude(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encode_shared(x)
        return self.state_encoder(h), self.amplitude_encoder(h)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z_s, _z_a = self.encode_state_amplitude(x)
        return z_s

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_s, z_a = self.encode_state_amplitude(x)
        return self.power_head(torch.cat([z_s, z_a], dim=-1))

    def _state_loss(self, state_logits: torch.Tensor, state_targets: torch.Tensor) -> torch.Tensor:
        pos_weight = None
        if self.state_pos_weight is not None:
            pos_weight = torch.as_tensor(self.state_pos_weight, dtype=state_logits.dtype, device=state_logits.device)
        return F.binary_cross_entropy_with_logits(state_logits, state_targets, pos_weight=pos_weight)

    def _state_conditioned_house_loss(
        self,
        z_s: torch.Tensor,
        state_targets: torch.Tensor,
        house_ids: torch.Tensor,
    ) -> torch.Tensor:
        house_logits = self.house_head(GradientReversal.apply(z_s, self.grl_lambda))
        losses = []
        binary_states = (state_targets >= 0.5)
        for state_value in (False, True):
            mask = binary_states == state_value
            if bool(mask.any()):
                losses.append(F.cross_entropy(house_logits[mask], house_ids[mask].long()))
        if not losses:
            return torch.zeros((), dtype=z_s.dtype, device=z_s.device)
        return torch.stack(losses).mean()

    def compute_loss(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        house_ids: torch.Tensor | None = None,
        criterion=None,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if criterion is None:
            criterion = nn.MSELoss()

        z_s, z_a = self.encode_state_amplitude(inputs)
        outputs = self.power_head(torch.cat([z_s, z_a], dim=-1))
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
                "state_conditioned_house_loss": zero.detach(),
            }

        state_targets = state_targets.to(device=inputs.device)
        state_logits = self.state_head(z_s).reshape(-1)
        self.last_state_logits = state_logits
        state_loss = self._state_loss(state_logits, state_targets)

        if house_ids is None:
            house_loss = zero
        else:
            house_loss = self._state_conditioned_house_loss(z_s, state_targets, house_ids.to(device=inputs.device))

        total_loss = power_loss + self.state_loss_weight * state_loss + self.house_loss_weight * house_loss
        if house_ids is None:
            return total_loss, outputs
        return total_loss, outputs, {
            "power_loss": power_loss.detach(),
            "state_loss": state_loss.detach(),
            "house_loss": house_loss.detach(),
            "state_conditioned_house_loss": house_loss.detach(),
        }

    def compute_selection_loss(self, inputs: torch.Tensor, targets: torch.Tensor, criterion) -> torch.Tensor:
        outputs = self.prepare_outputs(self(inputs))
        prepared_targets, _ = self._split_targets(targets)
        return criterion(outputs, prepared_targets)
