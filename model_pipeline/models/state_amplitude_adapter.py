"""Backbone-agnostic state-amplitude factorization adapter."""

from __future__ import annotations

import torch
import torch.nn as nn


class StateAmplitudeFactorizationAdapter(nn.Module):
    """Factor a global representation into state and conditional amplitude."""

    def __init__(
        self,
        *,
        input_dim: int,
        output_size: int,
        state_dim: int,
        amplitude_dim: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_size = int(output_size)
        self.state_dim = int(state_dim)
        self.amplitude_dim = int(amplitude_dim)

        self.state_encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.state_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.amplitude_encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.amplitude_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.state_head = nn.Linear(self.state_dim, self.output_size)
        self.amplitude_head = nn.Linear(self.amplitude_dim, self.output_size)

        # Targets use the common standardized appliance-power space.  The gate
        # therefore interpolates between a learned normalized off baseline and
        # the conditional on-amplitude prediction.
        self.off_level = nn.Parameter(torch.zeros(1))

    def encode_factors(
        self,
        representation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.state_encoder(representation),
            self.amplitude_encoder(representation),
        )

    def predict_from_factors(
        self,
        z_s: torch.Tensor,
        z_a: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state_logits = self.state_head(z_s)
        on_amplitude = self.amplitude_head(z_a)
        state_probability = torch.sigmoid(state_logits)
        power = self.off_level + state_probability * (
            on_amplitude - self.off_level
        )
        return power, state_logits, on_amplitude

    def forward(
        self,
        representation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z_s, z_a = self.encode_factors(representation)
        power, state_logits, on_amplitude = self.predict_from_factors(z_s, z_a)
        return power, state_logits, on_amplitude, z_s, z_a
