"""
Point-output BiLSTM baseline for NILM.

Reference source: https://github.com/nilmtk/nilmtk-contrib/blob/14efd545e09d836e02159b23444350af92c6ee70/nilmtk_contrib/torch/rnn.py
Implementation status: local bidirectional recurrent adaptation.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model_pipeline.model_registry import register_model
from model_pipeline.models.seq2point.rnn import RNNBaseline


@register_model("bilstm", aliases=("BiLSTM",), display_name="BiLSTM")
class BiLSTM(RNNBaseline):
    display_name = "BiLSTM"
    model_family = "rnn"

    default_window_size = 19
    default_num_epochs = 10
    official_batch_size = 512

    def __init__(self, *, window_size: int = 19, hidden_size: int = 128, num_layers: int = 2):
        nn.Module.__init__(self)
        if (window_size, hidden_size, num_layers) != (19, 128, 2):
            raise ValueError("BiLSTM uses its fixed reference defaults (19, 128, 2).")
        self.window_size = window_size
        self.output_size = 1
        self.output_offset = window_size // 2
        self._config = {
            "window_size": window_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
        }
        self.normalization = None
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
        return self.head(sequence_output[:, -1, :]).reshape(-1)
