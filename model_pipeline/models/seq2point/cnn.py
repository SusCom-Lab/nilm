"""
Seq2Point CNN baselines for NILM.

This file contains the Seq2Point baseline plus two project-level variants:
Seq2Point_Reduced and Seq2Point_Balanced.

These models are PyTorch adaptations of the sequence-to-point NILM family
proposed by Zhong et al. (2018). The canonical Seq2Point reference is:
https://github.com/MingjunZhong/seq2point-nilm

Citation:
    Zhong, M., Goddard, N., Sutton, C., and Gillian, C. (2018).
    "Sequence-to-point learning with neural networks for non-intrusive load
    monitoring." Proceedings of the AAAI Conference on Artificial Intelligence.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model_pipeline.model_registry import register_model
from model_pipeline.models.base_model import TorchNILMModel


@register_model(
    "original_seq2point",
    aliases=("Original Seq2Point", "seq2point_simple"),
    display_name="Seq2Point",
)
class OriginalSeq2Point(TorchNILMModel):
    display_name = "Seq2Point"
    model_family = "seq2point"
    target_type = "point"

    def __init__(self, *, window_size: int = 599, hidden_dim: int = 1024, **kwargs):
        super().__init__(window_size=window_size, hidden_dim=hidden_dim, **kwargs)
        self.features = nn.Sequential(
            nn.Conv1d(1, 30, kernel_size=10, padding="same"),
            nn.ReLU(),
            nn.Conv1d(30, 30, kernel_size=8, padding="same"),
            nn.ReLU(),
            nn.Conv1d(30, 40, kernel_size=6, padding="same"),
            nn.ReLU(),
            nn.Conv1d(40, 50, kernel_size=5, padding="same"),
            nn.ReLU(),
            nn.Conv1d(50, 50, kernel_size=5, padding="same"),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(50 * self.window_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.features(x)
        return self.head(x)


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
