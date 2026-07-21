"""
House-invariant state-aware BERT4NILM variant.

This file leaves the BERT4NILM baseline unchanged and adds the same auxiliary
state and house-invariant heads used by the Seq2Point/Seq2Seq variants.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_pipeline.model_registry import register_model
from model_pipeline.models.base_model import TorchNILMModel
from model_pipeline.models.seq2point.seq2point_invariant_state_aware import GradientReversal
from model_pipeline.models.seq2seq.bert4nilm import (
    BERT4NILMLayerNorm,
    PositionalEmbedding,
    TransformerBlock,
)


class _BERT4NILMBackbone(nn.Module):
    def __init__(self, *, window_size: int, hidden_dim: int, num_heads: int, num_layers: int, dropout: float):
        super().__init__()
        self.window_size = int(window_size)
        self.hidden = int(hidden_dim)
        self.latent_len = self.window_size // 2
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=self.hidden,
            kernel_size=5,
            stride=1,
            padding=2,
            padding_mode="replicate",
        )
        self.pool = nn.LPPool1d(norm_type=2, kernel_size=2, stride=2)
        self.position = PositionalEmbedding(max_len=self.latent_len, d_model=self.hidden)
        self.layer_norm = BERT4NILMLayerNorm(self.hidden)
        self.dropout = nn.Dropout(p=dropout)
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(self.hidden, num_heads, self.hidden * 4, dropout)
                for _ in range(int(num_layers))
            ]
        )
        self.deconv = nn.ConvTranspose1d(
            in_channels=self.hidden,
            out_channels=self.hidden,
            kernel_size=4,
            stride=2,
            padding=1,
        )

    def _match_window_length(self, x: torch.Tensor) -> torch.Tensor:
        current_len = x.size(1)
        if current_len == self.window_size:
            return x
        if current_len > self.window_size:
            return x[:, : self.window_size, :]
        return F.pad(x, (0, 0, 0, self.window_size - current_len))

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.conv(sequence.unsqueeze(1))).permute(0, 2, 1)
        x = x + self.position(sequence)
        x = self.dropout(self.layer_norm(x))
        for transformer in self.transformer_blocks:
            x = transformer(x, mask=None)
        x = self.deconv(x.permute(0, 2, 1)).permute(0, 2, 1)
        return self._match_window_length(x)


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
    "invariant_state_aware_bert4nilm",
    aliases=("InvariantStateAwareBERT4NILM", "isa_bert4nilm"),
    display_name="InvariantStateAwareBERT4NILM",
)
class InvariantStateAwareBERT4NILM(_SequenceStateMixin, TorchNILMModel):
    display_name = "InvariantStateAwareBERT4NILM"
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
        self.backbone = _BERT4NILMBackbone(
            window_size=self.window_size,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.power_linear1 = nn.Linear(hidden_dim, 128)
        self.power_linear2 = nn.Linear(128, 1)
        self.state_head = nn.Linear(hidden_dim, 1)
        self.house_head = nn.Linear(hidden_dim, self.num_domains)
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

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def _power_from_features(self, z: torch.Tensor) -> torch.Tensor:
        return self.power_linear2(torch.tanh(self.power_linear1(z))).squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._power_from_features(self.encode(x))

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
        outputs = self._power_from_features(z)
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
        state_logits = self.state_head(z).squeeze(-1)
        self.last_state_logits = state_logits
        state_loss = self._state_loss(state_logits, state_targets)

        if house_ids is None:
            house_loss = zero
        else:
            pooled_z = z.mean(dim=1)
            house_logits = self.house_head(GradientReversal.apply(pooled_z, self.grl_lambda))
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
