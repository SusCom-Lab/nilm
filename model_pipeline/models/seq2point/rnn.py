"""
Recurrent Seq2Point and point-output recurrent baselines for NILM.

This file contains the Seq2Point_LSTM variant used in this project, plus the
point-output recurrent baselines RNN, WindowGRU, and BiLSTM used in the shared
NILM benchmark. No confirmed public repository was identified as a direct
source port for the Seq2Point_LSTM variant. The recurrent baseline family is
aligned with:
https://github.com/nilmtk/nilmtk-contrib

Citation:
    Zhong, M., Goddard, N., Sutton, C., and Gillian, C. (2018).
    "Sequence-to-point learning with neural networks for non-intrusive load
    monitoring." Proceedings of the AAAI Conference on Artificial Intelligence.

    Kelly, J., and Knottenbelt, W. (2015).
    "Neural NILM: Deep Neural Networks Applied to Energy Disaggregation."
    Proceedings of the 2nd ACM International Conference on Embedded Systems
    for Energy-Efficient Built Environments.

    Batra, N., Kukunuri, R., Pandey, A., Malakar, R., Kumar, R.,
    Krystalakos, O., Zhong, M., Meira, P., and Parson, O. (2019).
    "Towards Reproducible State-of-the-Art Energy Disaggregation."
    Proceedings of the 6th ACM International Conference on Systems for
    Energy-Efficient Buildings, Cities, and Transportation.
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


@register_model(
    "legacy_lstm_seq2point",
    aliases=("LSTM-Based Seq2Point", "seq2point_lstm"),
    display_name="Seq2Point_LSTM",
)
class LegacyLSTMSeq2Point(TorchNILMModel):
    display_name = "Seq2Point_LSTM"
    model_family = "seq2point"
    target_type = "point"

    def __init__(self, *, window_size: int = 180, hidden_dim: int = 128, **kwargs):
        super().__init__(window_size=window_size, hidden_dim=hidden_dim, **kwargs)
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
        return self.head(x)


@register_model("rnn", aliases=("RNN",), display_name="RNN")
class RNNBaseline(_BasePointRNN):
    display_name = "RNN"
    model_family = "rnn"

    def __init__(self, *, window_size: int = 599, hidden_size: int = 128, num_layers: int = 2, **kwargs):
        super().__init__(window_size=window_size, hidden_size=hidden_size, num_layers=num_layers, **kwargs)
        self.encoder = nn.RNN(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity="tanh",
        )
        self.head = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)
        sequence_output, _ = self.encoder(x)
        return self._finalise(sequence_output)


@register_model("window_gru", aliases=("WindowGRU",), display_name="WindowGRU")
class WindowGRU(_BasePointRNN):
    display_name = "WindowGRU"
    model_family = "rnn"

    def __init__(self, *, window_size: int = 599, hidden_size: int = 128, num_layers: int = 2, **kwargs):
        super().__init__(window_size=window_size, hidden_size=hidden_size, num_layers=num_layers, **kwargs)
        self.encoder = nn.GRU(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)
        sequence_output, _ = self.encoder(x)
        return self._finalise(sequence_output)


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
