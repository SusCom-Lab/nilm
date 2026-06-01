"""
Balanced Seq2Point CNN variant for NILM.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model_pipeline.model_registry import register_model
from model_pipeline.models.base_model import TorchNILMModel


@register_model(
    "balanced_seq2point",
    aliases=("Balanced Seq2Point", "seq2point_balanced"),
    display_name="Seq2Point_Balanced",
)
class BalancedSeq2Point(TorchNILMModel):
    display_name = "Seq2Point_Balanced"
    model_family = "seq2point"
    target_type = "point"

    def __init__(self, *, window_size: int = 599, hidden_dim: int = 768, dropout: float = 0.3, **kwargs):
        super().__init__(window_size=window_size, hidden_dim=hidden_dim, dropout=dropout, **kwargs)
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=8, padding="same"),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 24, kernel_size=6, padding="same"),
            nn.BatchNorm1d(24),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(24, 32, kernel_size=5, padding="same"),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
        )
        pooled_length = max(1, self.window_size // 8)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * pooled_length, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.features(x)
        return self.head(x)
