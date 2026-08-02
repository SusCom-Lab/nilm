"""
Reduced Seq2Point CNN variant for NILM.

Base source: https://github.com/nilmtk/nilmtk-contrib/blob/14efd545e09d836e02159b23444350af92c6ee70/nilmtk_contrib/torch/seq2point.py
Implementation status: local reduced-capacity adaptation, not an upstream model.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model_pipeline.model_registry import register_model
from model_pipeline.models.seq2point.seq2point import Seq2Point


@register_model(
    "reduced_seq2point",
    aliases=("Dropout-Reduced Seq2Point", "seq2point_reduced"),
    display_name="Seq2Point_Reduced",
)
class ReducedSeq2Point(Seq2Point):
    display_name = "Seq2Point_Reduced"
    model_family = "seq2point"
    target_type = "point"

    default_window_size = 599
    default_num_epochs = 10

    def __init__(self, *, window_size: int = 599, hidden_dim: int = 512, dropout: float = 0.5):
        nn.Module.__init__(self)
        self.window_size = window_size
        self.output_size = 1
        self.output_offset = window_size // 2
        self._config = {"window_size": window_size, "hidden_dim": hidden_dim, "dropout": dropout}
        self.normalization = None
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
        return self.head(x).reshape(-1)
