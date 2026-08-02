"""
Point-output GRU baseline for NILM.

Official source: https://github.com/nilmtk/nilmtk-contrib/blob/14efd545e09d836e02159b23444350af92c6ee70/nilmtk_contrib/torch/WindowGRU.py
Local adaptation: point-output registry and plugin interfaces.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model_pipeline.model_registry import register_model
from model_pipeline.models.seq2point.rnn import _BasePointRNN


@register_model("window_gru", aliases=("WindowGRU",), display_name="WindowGRU")
class WindowGRU(_BasePointRNN):
    display_name = "WindowGRU"
    model_family = "rnn"

    def __init__(self, *, window_size: int = 599, **kwargs):
        super().__init__(window_size=window_size, **kwargs)
        self.conv1 = nn.Conv1d(1, 16, kernel_size=4, padding=2)
        self.gru1 = nn.GRU(16, 64, batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(0.5)
        self.gru2 = nn.GRU(128, 128, batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256, 128)
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = torch.relu(self.conv1(x))
        x = x.permute(0, 2, 1)
        x, _ = self.gru1(x)
        x = self.dropout1(x)
        _, hidden = self.gru2(x)
        x = torch.cat([hidden[-2], hidden[-1]], dim=1)
        x = self.dropout2(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout3(x)
        return self.fc2(x)
