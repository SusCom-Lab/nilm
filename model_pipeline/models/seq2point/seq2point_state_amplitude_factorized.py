"""Seq2Point with the minimal state-amplitude factorization adapter.

The original Seq2Point CNN/flatten/hidden representation is inherited without
change.  Only its final hidden-to-power layer is replaced by the shared
state-amplitude adapter.  The objective contains power, state, and conditional
on-amplitude losses; it intentionally contains no house/domain loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_pipeline.model_registry import register_model
from model_pipeline.models.seq2point.seq2point import Seq2Point
from model_pipeline.models.state_amplitude_adapter import (
    StateAmplitudeFactorizationAdapter,
)


@register_model(
    "state_amplitude_factorized_seq2point",
    aliases=("StateAmplitudeFactorizedSeq2Point", "saf_seq2point"),
    display_name="StateAmplitudeFactorizedSeq2Point",
)
class StateAmplitudeFactorizedSeq2Point(Seq2Point):
    """Original Seq2Point global backbone plus the shared factorization adapter."""

    display_name = "StateAmplitudeFactorizedSeq2Point"

    def __init__(
        self,
        *,
        window_size: int = 599,
        hidden_dim: int = 1024,
        state_dim: int | None = None,
        amplitude_dim: int | None = None,
        state_loss_weight: float = 0.1,
        amplitude_loss_weight: float = 0.05,
        state_pos_weight: float | None = None,
        amplitude_loss: str = "smooth_l1",
        selection_objective: str = "power",
        **kwargs,
    ) -> None:
        state_dim = int(state_dim if state_dim is not None else hidden_dim // 2)
        amplitude_dim = int(
            amplitude_dim if amplitude_dim is not None else hidden_dim - state_dim
        )
        if state_dim <= 0 or amplitude_dim <= 0:
            raise ValueError("state_dim and amplitude_dim must both be positive.")
        if state_dim + amplitude_dim != int(hidden_dim):
            raise ValueError("state_dim + amplitude_dim must equal hidden_dim.")
        if amplitude_loss not in {"l1", "mse", "smooth_l1"}:
            raise ValueError("amplitude_loss must be 'l1', 'mse', or 'smooth_l1'.")
        if selection_objective not in {"power", "total"}:
            raise ValueError("selection_objective must be 'power' or 'total'.")

        super().__init__(
            window_size=window_size,
            hidden_dim=hidden_dim,
            state_dim=state_dim,
            amplitude_dim=amplitude_dim,
            state_loss_weight=float(state_loss_weight),
            amplitude_loss_weight=float(amplitude_loss_weight),
            state_pos_weight=state_pos_weight,
            amplitude_loss=amplitude_loss,
            selection_objective=selection_objective,
            **kwargs,
        )
        if self.output_size != 1:
            raise ValueError("StateAmplitudeFactorizedSeq2Point requires output_size=1.")

        self.hidden_dim = int(hidden_dim)
        self.state_dim = state_dim
        self.amplitude_dim = amplitude_dim
        self.state_loss_weight = float(state_loss_weight)
        self.amplitude_loss_weight = float(amplitude_loss_weight)
        self.state_pos_weight = (
            None if state_pos_weight is None else float(state_pos_weight)
        )
        self.amplitude_loss = amplitude_loss
        self.selection_objective = selection_objective

        # Seq2Point.network[-1] is the baseline Linear(hidden_dim, 1) head.
        # Replacing only that layer preserves every backbone operation and its
        # ordering while exposing the original hidden representation h.
        self.network[-1] = nn.Identity()
        self.factorization_adapter = StateAmplitudeFactorizationAdapter(
            input_dim=self.hidden_dim,
            output_size=1,
            state_dim=self.state_dim,
            amplitude_dim=self.amplitude_dim,
            dropout=0.2,
        )
        self.last_state_logits: torch.Tensor | None = None
        self.last_on_amplitude: torch.Tensor | None = None
        self.last_loss_components: dict[str, torch.Tensor] = {}

    def prepare_targets(self, targets):
        if (
            isinstance(targets, torch.Tensor)
            and targets.ndim == 2
            and targets.size(-1) == 2
        ):
            return targets.float()
        return super().prepare_targets(targets)

    def _split_targets(
        self,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if targets.ndim == 2 and targets.size(-1) == 2:
            return targets[:, 0].reshape(-1).float(), targets[:, 1].reshape(-1).float()
        return super().prepare_targets(targets), None

    def encode_shared(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x.unsqueeze(1))

    def forward_with_factors(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.factorization_adapter(self.encode_shared(x))

    def encode_state_amplitude(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.factorization_adapter.encode_factors(self.encode_shared(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z_s, _ = self.encode_state_amplitude(x)
        return z_s

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        power, _, _, _, _ = self.forward_with_factors(x)
        return power

    def _state_loss(
        self,
        state_logits: torch.Tensor,
        state_targets: torch.Tensor,
    ) -> torch.Tensor:
        logits = state_logits.reshape(-1)
        pos_weight = None
        if self.state_pos_weight is not None:
            pos_weight = torch.as_tensor(
                self.state_pos_weight,
                dtype=logits.dtype,
                device=logits.device,
            )
        return F.binary_cross_entropy_with_logits(
            logits,
            state_targets.reshape(-1),
            pos_weight=pos_weight,
        )

    def _conditional_amplitude_loss(
        self,
        on_amplitude: torch.Tensor,
        power_targets: torch.Tensor,
        state_targets: torch.Tensor,
    ) -> torch.Tensor:
        amplitude = on_amplitude.reshape(-1)
        power = power_targets.reshape(-1)
        if self.amplitude_loss == "l1":
            elementwise = F.l1_loss(amplitude, power, reduction="none")
        elif self.amplitude_loss == "mse":
            elementwise = F.mse_loss(amplitude, power, reduction="none")
        else:
            elementwise = F.smooth_l1_loss(amplitude, power, reduction="none")

        mask = state_targets.reshape(-1).to(dtype=elementwise.dtype)
        positive_count = mask.sum()
        if not bool(positive_count > 0):
            return elementwise.sum() * 0.0
        return (elementwise * mask).sum() / positive_count

    def _loss_components(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        criterion,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        outputs, state_logits, on_amplitude, _, _ = self.forward_with_factors(inputs)
        power_targets, state_targets = self._split_targets(targets)
        power_loss = criterion(self.prepare_outputs(outputs), power_targets)
        zero = power_loss.new_zeros(())

        if state_targets is None:
            return outputs, power_loss, zero, zero, power_loss

        state_targets = state_targets.to(device=inputs.device)
        state_loss = self._state_loss(state_logits, state_targets)
        amplitude_loss = self._conditional_amplitude_loss(
            on_amplitude,
            power_targets,
            state_targets,
        )
        total_loss = (
            power_loss
            + self.state_loss_weight * state_loss
            + self.amplitude_loss_weight * amplitude_loss
        )
        self.last_state_logits = state_logits
        self.last_on_amplitude = on_amplitude
        return outputs, power_loss, state_loss, amplitude_loss, total_loss

    def compute_loss(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        criterion=None,
    ):
        if criterion is None:
            criterion = nn.MSELoss()
        outputs, power_loss, state_loss, amplitude_loss, total_loss = (
            self._loss_components(inputs, targets, criterion)
        )
        self.last_loss_components = {
            "power_loss": power_loss.detach(),
            "state_loss": state_loss.detach(),
            "amplitude_loss": amplitude_loss.detach(),
        }
        return total_loss, outputs

    def compute_selection_loss(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        criterion,
    ) -> torch.Tensor:
        _, power_loss, _, _, total_loss = self._loss_components(
            inputs,
            targets,
            criterion,
        )
        if self.selection_objective == "total":
            return total_loss
        return power_loss
