"""
State-conditioned invariant BERT4NILM variant.

This model separates BERT4NILM sequence features into z_s and z_a. State
supervision and state-conditioned house invariance are applied to z_s, while
power prediction uses [z_s, z_a].
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_pipeline.model_registry import register_model
from model_pipeline.models.base_model import TorchNILMModel
from model_pipeline.models.seq2point.seq2point_invariant_state_aware import GradientReversal
from model_pipeline.models.seq2seq.bert4nilm_invariant_state_aware import (
    _BERT4NILMBackbone,
    _SequenceStateMixin,
)


@register_model(
    "state_conditioned_invariant_bert4nilm",
    aliases=("StateConditionedInvariantBERT4NILM", "sci_bert4nilm"),
    display_name="StateConditionedInvariantBERT4NILM",
)
class StateConditionedInvariantBERT4NILM(_SequenceStateMixin, TorchNILMModel):
    display_name = "StateConditionedInvariantBERT4NILM"
    model_family = "transformer"
    target_type = "sequence"
    default_output_size = None

    def __init__(
        self,
        *,
        window_size: int = 599,
        hidden_dim: int = 256,
        num_heads: int = 2,
        num_layers: int = 2,
        dropout: float = 0.1,
        state_dim: int | None = None,
        amplitude_dim: int | None = None,
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
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            state_dim=state_dim,
            amplitude_dim=amplitude_dim,
            state_loss_weight=state_loss_weight,
            state_pos_weight=state_pos_weight,
            num_domains=num_domains,
            house_loss_weight=house_loss_weight,
            grl_lambda=grl_lambda,
            **kwargs,
        )
        if state_dim is None and amplitude_dim is None:
            state_dim = hidden_dim // 2
            amplitude_dim = hidden_dim - state_dim
        elif state_dim is None:
            state_dim = hidden_dim - int(amplitude_dim)
        elif amplitude_dim is None:
            amplitude_dim = hidden_dim - int(state_dim)
        self.state_dim = int(state_dim)
        self.amplitude_dim = int(amplitude_dim)
        if self.state_dim <= 0 or self.amplitude_dim <= 0:
            raise ValueError("state_dim and amplitude_dim must both be positive.")

        self.state_loss_weight = float(state_loss_weight)
        self.state_pos_weight = None if state_pos_weight is None else float(state_pos_weight)
        self.num_domains = int(num_domains)
        self.house_loss_weight = float(house_loss_weight)
        self.grl_lambda = float(grl_lambda)

        self.backbone = _BERT4NILMBackbone(
            window_size=self.window_size,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(hidden_dim, self.state_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.amplitude_encoder = nn.Sequential(
            nn.Linear(hidden_dim, self.amplitude_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.power_linear1 = nn.Linear(self.state_dim + self.amplitude_dim, 128)
        self.power_linear2 = nn.Linear(128, 1)
        self.state_head = nn.Linear(self.state_dim, 1)
        self.house_head = nn.Linear(self.state_dim, self.num_domains)
        self.last_state_logits: torch.Tensor | None = None
        self.truncated_normal_init()

    def truncated_normal_init(self, mean: float = 0.0, std: float = 0.02, lower: float = -0.04, upper: float = 0.04):
        for name, parameter in self.named_parameters():
            if "layer_norm" in name:
                continue
            with torch.no_grad():
                l_value = (1.0 + math.erf(((lower - mean) / std) / math.sqrt(2.0))) / 2.0
                u_value = (1.0 + math.erf(((upper - mean) / std) / math.sqrt(2.0))) / 2.0
                parameter.uniform_(2 * l_value - 1, 2 * u_value - 1)
                parameter.erfinv_()
                parameter.mul_(std * math.sqrt(2.0))
                parameter.add_(mean)

    def encode_shared(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def encode_state_amplitude(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encode_shared(x)
        return self.state_encoder(h), self.amplitude_encoder(h)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z_s, _z_a = self.encode_state_amplitude(x)
        return z_s

    def _power_from_features(self, z_s: torch.Tensor, z_a: torch.Tensor) -> torch.Tensor:
        features = torch.cat([z_s, z_a], dim=-1)
        return self.power_linear2(torch.tanh(self.power_linear1(features))).squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_s, z_a = self.encode_state_amplitude(x)
        return self._power_from_features(z_s, z_a)

    def _state_conditioned_house_loss(
        self,
        z_s: torch.Tensor,
        state_targets: torch.Tensor,
        house_ids: torch.Tensor,
    ) -> torch.Tensor:
        window_states = (state_targets >= 0.5).any(dim=1)
        pooled_z = z_s.mean(dim=1)
        house_logits = self.house_head(GradientReversal.apply(pooled_z, self.grl_lambda))
        losses = []
        for state_value in (False, True):
            mask = window_states == state_value
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
    ):
        if criterion is None:
            criterion = nn.MSELoss()
        z_s, z_a = self.encode_state_amplitude(inputs)
        outputs = self._power_from_features(z_s, z_a)
        power_targets, state_targets = self._split_targets(targets)
        power_loss = criterion(self.prepare_outputs(outputs), power_targets)

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
        state_logits = self.state_head(z_s).squeeze(-1)
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
        power_targets, _ = self._split_targets(targets)
        return criterion(outputs, power_targets)
