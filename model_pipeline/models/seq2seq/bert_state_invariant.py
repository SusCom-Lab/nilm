"""
State-aware and state-conditioned invariant variants of the local BERT model.

These models extend model_pipeline.models.seq2seq.bert.BERTNILM, which is a
BERT-inspired sequence-to-sequence NILM baseline. They are intentionally kept
separate from BERT4NILM.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_pipeline.model_registry import register_model
from model_pipeline.models.base_model import TorchNILMModel
from model_pipeline.models.seq2seq.bert import LPpool, PositionalEncoding, TransformerBlock
from model_pipeline.models.seq2point.seq2point_invariant_state_aware import GradientReversal


class _BERTFeatureEncoder(nn.Module):
    def __init__(self, *, window_size: int, embed_dim: int, num_heads: int, ff_dim: int):
        super().__init__()
        self.window_size = int(window_size)
        self.embed_dim = int(embed_dim)
        self.pooled_length = ((self.window_size - 2) // 2) + 1
        self.conv = nn.Conv1d(1, embed_dim, 4, stride=1, padding="same")
        self.pool = LPpool(pool_size=2)
        self.position = PositionalEncoding(embed_dim, self.pooled_length)
        self.transformer = TransformerBlock(embed_dim, num_heads, ff_dim)
        self.flatten = nn.Flatten()

    @property
    def output_dim(self) -> int:
        return self.pooled_length * self.embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1).permute(0, 2, 1)
        x = self.conv(x)
        x = self.pool(x)
        x = x.permute(0, 2, 1)
        x = self.position(x)
        x = self.transformer(x)
        return self.flatten(x)


class _SequenceStateMixin:
    def prepare_targets(self, targets):
        if isinstance(targets, torch.Tensor) and targets.ndim == 3 and targets.size(-1) == 2:
            return targets.float()
        return super().prepare_targets(targets)

    def _split_targets(self, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        if targets.ndim == 3 and targets.size(-1) == 2:
            return targets[..., 0].float(), targets[..., 1].float()
        return super().prepare_targets(targets), None

    def _state_loss(self, state_logits: torch.Tensor, state_targets: torch.Tensor) -> torch.Tensor:
        pos_weight = None
        if self.state_pos_weight is not None:
            pos_weight = torch.as_tensor(self.state_pos_weight, dtype=state_logits.dtype, device=state_logits.device)
        return F.binary_cross_entropy_with_logits(state_logits, state_targets, pos_weight=pos_weight)


@register_model(
    "invariant_state_aware_bert",
    aliases=("InvariantStateAwareBERT", "isa_bert"),
    display_name="InvariantStateAwareBERT",
)
class InvariantStateAwareBERT(_SequenceStateMixin, TorchNILMModel):
    display_name = "InvariantStateAwareBERT"
    model_family = "transformer"
    target_type = "sequence"
    default_output_size = None

    def __init__(
        self,
        *,
        window_size: int = 99,
        embed_dim: int = 32,
        num_heads: int = 2,
        ff_dim: int = 32,
        state_loss_weight: float = 0.05,
        state_pos_weight: float | None = None,
        num_domains: int = 2,
        house_loss_weight: float = 0.003,
        grl_lambda: float = 1.0,
        **kwargs,
    ):
        super().__init__(
            window_size=window_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            state_loss_weight=state_loss_weight,
            state_pos_weight=state_pos_weight,
            num_domains=num_domains,
            house_loss_weight=house_loss_weight,
            grl_lambda=grl_lambda,
            **kwargs,
        )
        self.state_loss_weight = float(state_loss_weight)
        self.state_pos_weight = None if state_pos_weight is None else float(state_pos_weight)
        self.num_domains = int(num_domains)
        self.house_loss_weight = float(house_loss_weight)
        self.grl_lambda = float(grl_lambda)

        self.encoder = _BERTFeatureEncoder(
            window_size=self.window_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
        )
        feature_dim = self.encoder.output_dim
        self.dropout = nn.Dropout(0.1)
        self.power_head = nn.Linear(feature_dim, self.output_size)
        self.power_dropout = nn.Dropout(0.1)
        self.state_head = nn.Linear(feature_dim, self.output_size)
        self.house_head = nn.Linear(feature_dim, self.num_domains)
        self.last_state_logits: torch.Tensor | None = None

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.encoder(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.power_dropout(self.power_head(self.encode(x)))

    def compute_loss(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        house_ids: torch.Tensor | None = None,
        criterion=None,
    ):
        if criterion is None:
            criterion = nn.MSELoss()
        z = self.encode(inputs)
        outputs = self.power_dropout(self.power_head(z))
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
            }

        state_targets = state_targets.to(device=inputs.device)
        state_logits = self.state_head(z)
        self.last_state_logits = state_logits
        state_loss = self._state_loss(state_logits, state_targets)

        if house_ids is None:
            house_loss = zero
        else:
            house_logits = self.house_head(GradientReversal.apply(z, self.grl_lambda))
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
        power_targets, _ = self._split_targets(targets)
        return criterion(outputs, power_targets)


@register_model(
    "state_conditioned_invariant_bert",
    aliases=("StateConditionedInvariantBERT", "sci_bert"),
    display_name="StateConditionedInvariantBERT",
)
class StateConditionedInvariantBERT(_SequenceStateMixin, TorchNILMModel):
    display_name = "StateConditionedInvariantBERT"
    model_family = "transformer"
    target_type = "sequence"
    default_output_size = None

    def __init__(
        self,
        *,
        window_size: int = 99,
        embed_dim: int = 32,
        num_heads: int = 2,
        ff_dim: int = 32,
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
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            state_dim=state_dim,
            amplitude_dim=amplitude_dim,
            state_loss_weight=state_loss_weight,
            state_pos_weight=state_pos_weight,
            num_domains=num_domains,
            house_loss_weight=house_loss_weight,
            grl_lambda=grl_lambda,
            **kwargs,
        )
        self.state_loss_weight = float(state_loss_weight)
        self.state_pos_weight = None if state_pos_weight is None else float(state_pos_weight)
        self.num_domains = int(num_domains)
        self.house_loss_weight = float(house_loss_weight)
        self.grl_lambda = float(grl_lambda)

        self.encoder = _BERTFeatureEncoder(
            window_size=self.window_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
        )
        feature_dim = self.encoder.output_dim
        if state_dim is None and amplitude_dim is None:
            state_dim = feature_dim // 2
            amplitude_dim = feature_dim - state_dim
        elif state_dim is None:
            state_dim = feature_dim - int(amplitude_dim)
        elif amplitude_dim is None:
            amplitude_dim = feature_dim - int(state_dim)
        self.state_dim = int(state_dim)
        self.amplitude_dim = int(amplitude_dim)
        if self.state_dim <= 0 or self.amplitude_dim <= 0:
            raise ValueError("state_dim and amplitude_dim must both be positive.")

        self.shared_dropout = nn.Dropout(0.1)
        self.state_encoder = nn.Sequential(
            nn.Linear(feature_dim, self.state_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.amplitude_encoder = nn.Sequential(
            nn.Linear(feature_dim, self.amplitude_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.power_head = nn.Linear(self.state_dim + self.amplitude_dim, self.output_size)
        self.power_dropout = nn.Dropout(0.1)
        self.state_head = nn.Linear(self.state_dim, self.output_size)
        self.house_head = nn.Linear(self.state_dim, self.num_domains)
        self.last_state_logits: torch.Tensor | None = None

    def encode_shared(self, x: torch.Tensor) -> torch.Tensor:
        return self.shared_dropout(self.encoder(x))

    def encode_state_amplitude(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encode_shared(x)
        return self.state_encoder(h), self.amplitude_encoder(h)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z_s, _z_a = self.encode_state_amplitude(x)
        return z_s

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_s, z_a = self.encode_state_amplitude(x)
        return self.power_dropout(self.power_head(torch.cat([z_s, z_a], dim=-1)))

    def _state_conditioned_house_loss(
        self,
        z_s: torch.Tensor,
        state_targets: torch.Tensor,
        house_ids: torch.Tensor,
    ) -> torch.Tensor:
        house_logits = self.house_head(GradientReversal.apply(z_s, self.grl_lambda))
        window_states = (state_targets >= 0.5).any(dim=1)
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
        outputs = self.power_dropout(self.power_head(torch.cat([z_s, z_a], dim=-1)))
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
        state_logits = self.state_head(z_s)
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
