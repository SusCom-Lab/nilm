"""
LSTM-based Seq2Point variant for NILM.

Reference source: https://github.com/nilmtk/nilmtk-contrib/blob/14efd545e09d836e02159b23444350af92c6ee70/nilmtk_contrib/torch/rnn.py
Implementation status: local sequence-to-point adaptation.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model_pipeline.model_registry import register_model
from model_pipeline.models.seq2point.seq2point import Seq2Point


@register_model(
    "legacy_lstm_seq2point",
    aliases=("LSTM-Based Seq2Point", "seq2point_lstm"),
    display_name="Seq2Point_LSTM",
)
class LegacyLSTMSeq2Point(Seq2Point):
    display_name = "Seq2Point_LSTM"
    model_family = "seq2point"
    target_type = "point"

    default_window_size = 180
    default_num_epochs = 10

    def __init__(self, *, window_size: int = 180, hidden_dim: int = 128):
        nn.Module.__init__(self)
        self.window_size = window_size
        self.output_size = 1
        self.output_offset = window_size // 2
        self._config = {"window_size": window_size, "hidden_dim": hidden_dim}
        self.normalization = None
        self.pad = nn.ConstantPad1d((1, 2), 0.0)
        self.conv = nn.Conv1d(1, 16, kernel_size=4)
        self.lstm1 = nn.LSTM(input_size=16, hidden_size=64, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=128, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.Linear(self.window_size * 256, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.pad(x)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x.reshape(x.size(0), -1)
        return self.head(x).reshape(-1)
