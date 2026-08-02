"""
Reduced Seq2Point CNN variant for NILM.

Base source: https://github.com/nilmtk/nilmtk-contrib/blob/14efd545e09d836e02159b23444350af92c6ee70/nilmtk_contrib/torch/seq2point.py
Implementation status: local reduced-capacity adaptation, not an upstream model.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model_pipeline.model_registry import register_model
from model_pipeline.models.base_model import TorchNILMModel


@register_model(
    "reduced_seq2point",
    aliases=("Dropout-Reduced Seq2Point", "seq2point_reduced"),
    display_name="Seq2Point_Reduced",
)
class ReducedSeq2Point(TorchNILMModel):
    display_name = "Seq2Point_Reduced"
    model_family = "seq2point"
    target_type = "point"

    def __init__(self, *, window_size: int = 599, hidden_dim: int = 512, dropout: float = 0.5, **kwargs):
        super().__init__(window_size=window_size, hidden_dim=hidden_dim, dropout=dropout, **kwargs)
        self.features = nn.Sequential(
            nn.Conv1d(1, 20, kernel_size=8, padding="same"),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(20, 20, kernel_size=6, padding="same"),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(20, 30, kernel_size=5, padding="same"),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(30, 40, kernel_size=4, padding="same"),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(40, 40, kernel_size=4, padding="same"),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(40 * self.window_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.features(x)
        return self.head(x)
