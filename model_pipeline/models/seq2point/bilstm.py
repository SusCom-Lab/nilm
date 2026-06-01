"""
Point-output BiLSTM baseline for NILM.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model_pipeline.model_registry import register_model
from model_pipeline.models.seq2point.rnn import _BasePointRNN


@register_model("bilstm", aliases=("BiLSTM",), display_name="BiLSTM")
class BiLSTM(_BasePointRNN):
    display_name = "BiLSTM"
    model_family = "rnn"

    def __init__(self, *, window_size: int = 599, hidden_size: int = 128, num_layers: int = 2, **kwargs):
        super().__init__(window_size=window_size, hidden_size=hidden_size, num_layers=num_layers, **kwargs)
        self.encoder = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)
        sequence_output, _ = self.encoder(x)
        return self._finalise(sequence_output)
