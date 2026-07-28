"""Aggregate-only encoder used by Context-FiLM."""

from __future__ import annotations

import torch
import torch.nn as nn


class ContextEncoder(nn.Module):
    """Encode each aggregate window and masked-mean pool a household code."""

    def __init__(self, window_size: int = 599, code_dim: int = 128):
        super().__init__()
        self.window_size = int(window_size)
        self.code_dim = int(code_dim)
        self.window_encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=9, stride=2, padding=4),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, self.code_dim),
        )

    def forward(
        self,
        windows: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if windows.ndim == 2:
            windows = windows.unsqueeze(0)
        if windows.ndim != 3 or windows.shape[-1] != self.window_size:
            raise ValueError(
                f"Expected [batch, context_k, {self.window_size}] context windows, "
                f"got {tuple(windows.shape)}."
            )
        batch, context_k, _ = windows.shape
        encoded = self.window_encoder(windows.reshape(batch * context_k, 1, self.window_size))
        encoded = encoded.reshape(batch, context_k, self.code_dim)
        if mask is None:
            mask = torch.ones(batch, context_k, dtype=torch.bool, device=windows.device)
        if mask.shape != (batch, context_k):
            raise ValueError(f"Expected mask shape {(batch, context_k)}, got {tuple(mask.shape)}.")
        weights = mask.to(encoded.dtype).unsqueeze(-1)
        denominator = weights.sum(dim=1).clamp_min(1.0)
        return (encoded * weights).sum(dim=1) / denominator
