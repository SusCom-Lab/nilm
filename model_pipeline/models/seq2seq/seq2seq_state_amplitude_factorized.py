"""Seq2Seq with a minimal state-amplitude factorization adapter.

The original Seq2Seq CNN/flatten/FC representation is left unchanged.  The
adapter is deliberately small and reusable: it maps the global representation
to a state factor and an on-amplitude factor, then reconstructs power with a
soft state gate.  Training uses only power, state, and optional conditional
amplitude supervision.  No domain adversary or auxiliary disentanglement
regularizers are part of this model.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_pipeline.model_registry import register_model
from model_pipeline.models.state_amplitude_adapter import (
    StateAmplitudeFactorizationAdapter,
)
from model_pipeline.models.seq2seq.seq2seq import Seq2SeqCNN


@register_model(
    "state_amplitude_factorized_seq2seq",
    aliases=("StateAmplitudeFactorizedSeq2Seq", "saf_seq2seq"),
    display_name="StateAmplitudeFactorizedSeq2Seq",
)
class StateAmplitudeFactorizedSeq2Seq(Seq2SeqCNN):
    """Original global Seq2Seq backbone plus the factorization adapter."""

    display_name = "StateAmplitudeFactorizedSeq2Seq"

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

        self.state_dim = state_dim
        self.amplitude_dim = amplitude_dim
        self.state_loss_weight = float(state_loss_weight)
        self.amplitude_loss_weight = float(amplitude_loss_weight)
        self.state_pos_weight = (
            None if state_pos_weight is None else float(state_pos_weight)
        )
        self.amplitude_loss = amplitude_loss
        self.selection_objective = selection_objective

        # The baseline output head is replaced, while conv1..conv5, flatten,
        # fc1, and all dropout layers remain exactly those of Seq2SeqCNN.
        self.fc2 = nn.Identity()
        self.factorization_adapter = StateAmplitudeFactorizationAdapter(
            input_dim=self.hidden_dim,
            output_size=self.output_size,
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
            and targets.ndim == 3
            and targets.size(-1) == 2
        ):
            return targets.float()
        return super().prepare_targets(targets)

    def _split_targets(
        self,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if targets.ndim == 3 and targets.size(-1) == 2:
            return targets[..., 0].float(), targets[..., 1].float()
        return super().prepare_targets(targets), None

    def forward_with_factors(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.factorization_adapter(super().encode(x))

    def encode_state_amplitude(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        representation = super().encode(x)
        return self.factorization_adapter.encode_factors(representation)

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
        pos_weight = None
        if self.state_pos_weight is not None:
            pos_weight = torch.as_tensor(
                self.state_pos_weight,
                dtype=state_logits.dtype,
                device=state_logits.device,
            )
        return F.binary_cross_entropy_with_logits(
            state_logits,
            state_targets,
            pos_weight=pos_weight,
        )

    def _conditional_amplitude_loss(
        self,
        on_amplitude: torch.Tensor,
        power_targets: torch.Tensor,
        state_targets: torch.Tensor,
    ) -> torch.Tensor:
        if self.amplitude_loss == "l1":
            elementwise = F.l1_loss(on_amplitude, power_targets, reduction="none")
        elif self.amplitude_loss == "mse":
            elementwise = F.mse_loss(on_amplitude, power_targets, reduction="none")
        else:
            elementwise = F.smooth_l1_loss(
                on_amplitude,
                power_targets,
                reduction="none",
            )

        mask = state_targets.to(dtype=elementwise.dtype)
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
