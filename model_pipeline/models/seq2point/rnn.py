"""
Point-output RNN baseline for NILM.

Official source: https://github.com/nilmtk/nilmtk-contrib/blob/14efd545e09d836e02159b23444350af92c6ee70/nilmtk_contrib/torch/rnn.py
Local adaptation: point-output registry and plugin interfaces.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model_pipeline.model_registry import register_model
from model_pipeline.models.base_model import TorchNILMModel


class _BasePointRNN(TorchNILMModel):
    target_type = "point"

    def _finalise(self, sequence_output: torch.Tensor) -> torch.Tensor:
        last_hidden = sequence_output[:, -1, :]
        return self.head(last_hidden)


@register_model("rnn", aliases=("RNN",), display_name="RNN")
class RNNBaseline(_BasePointRNN):
    display_name = "RNN"
    model_family = "rnn"

    def __init__(self, *, window_size: int = 599, **kwargs):
        super().__init__(window_size=window_size, **kwargs)
        self.conv1d = nn.Conv1d(
            in_channels=1,
            out_channels=16,
            kernel_size=4,
            stride=1,
            padding=2,
        )
        self.lstm1 = nn.LSTM(
            input_size=16,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.lstm2 = nn.LSTM(
            input_size=256,
            hidden_size=256,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1).permute(0, 2, 1)
        x = self.conv1d(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm1(x)
        x = self.dropout(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]
        x = torch.tanh(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)
