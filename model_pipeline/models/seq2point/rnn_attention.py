"""
RNN-attention NILM model matching nilmtk-contrib architecture.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_pipeline.model_registry import register_model
from model_pipeline.models.base_model import TorchNILMModel


class AttentionLayer(nn.Module):
    def __init__(self, units: int):
        super().__init__()
        self.W = nn.Linear(512, units)
        self.V = nn.Linear(units, 1)
        nn.init.kaiming_normal_(self.W.weight, mode="fan_in", nonlinearity="relu")
        nn.init.kaiming_normal_(self.V.weight, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(self.W.bias)
        nn.init.zeros_(self.V.bias)

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        score = self.V(torch.tanh(self.W(encoder_output)))
        attention_weights = F.softmax(score, dim=1)
        context_vector = attention_weights * encoder_output
        return torch.sum(context_vector, dim=1)


@register_model("rnn_attention", aliases=("RNN_attention", "RNNAttention"), display_name="RNN_Attention")
class RNNAttentionNILM(TorchNILMModel):
    display_name = "RNN_Attention"
    model_family = "rnn_attention"
    target_type = "point"

    def __init__(self, *, window_size: int = 19, **kwargs):
        super().__init__(window_size=window_size, **kwargs)
        self.conv1d = nn.Conv1d(1, 16, kernel_size=4, stride=1, padding=2)
        self.lstm1 = nn.LSTM(16, 128, num_layers=1, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(256, 256, num_layers=1, batch_first=True, bidirectional=True)
        self.attention = AttentionLayer(units=128)
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
        x = self.attention(x)
        x = torch.tanh(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)
