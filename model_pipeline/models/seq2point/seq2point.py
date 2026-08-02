"""
Seq2Point CNN baseline for NILM.

Official source: https://github.com/nilmtk/nilmtk-contrib/blob/14efd545e09d836e02159b23444350af92c6ee70/nilmtk_contrib/torch/seq2point.py
Local adaptation: registry and repository plugin interfaces.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model_pipeline.model_registry import register_model
from model_pipeline.models.base_model import TorchNILMModel


@register_model(
    "seq2point",
    aliases=("seq2point_simple",),
    display_name="Seq2Point",
)
class Seq2Point(TorchNILMModel):
    display_name = "Seq2Point"
    model_family = "seq2point"
    target_type = "point"

    def __init__(self, *, window_size: int = 599, hidden_dim: int = 1024, **kwargs):
        super().__init__(window_size=window_size, hidden_dim=hidden_dim, **kwargs)
        conv_reduction = (10 - 1) + (8 - 1) + (6 - 1) + (5 - 1) + (5 - 1)
        flattened = 50 * (self.window_size - conv_reduction)
        self.network = nn.Sequential(
            nn.Conv1d(1, 30, kernel_size=10, stride=1),
            nn.ReLU(),
            nn.Conv1d(30, 30, kernel_size=8, stride=1),
            nn.ReLU(),
            nn.Conv1d(30, 40, kernel_size=6, stride=1),
            nn.ReLU(),
            nn.Conv1d(40, 50, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(50, 50, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Flatten(),
            nn.Linear(flattened, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the last hidden representation without changing legacy weights."""
        x = x.unsqueeze(1)
        return self.network[:-1](x)

    def decode(self, hidden: torch.Tensor) -> torch.Tensor:
        """Apply the original output layer to a hidden representation."""
        return self.network[-1](hidden)
