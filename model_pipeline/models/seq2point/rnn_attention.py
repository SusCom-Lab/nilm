"""
RNN-attention NILM model matching nilmtk-contrib architecture.

Official source: https://github.com/nilmtk/nilmtk-contrib/blob/14efd545e09d836e02159b23444350af92c6ee70/nilmtk_contrib/torch/rnn_attention.py
Local adaptation: repository registry and plugin interfaces.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_pipeline.model_registry import register_model
from model_pipeline.models.seq2point.rnn import RNNBaseline


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
class RNNAttentionNILM(RNNBaseline):
    display_name = "RNN_Attention"
    model_family = "rnn_attention"
    target_type = "point"

    default_window_size = 19
    default_num_epochs = 10
    official_batch_size = 512
    official_optimizer_epsilon = 1e-8
    official_gradient_clip_norm = 1.0

    def __init__(self, *, window_size: int = 19):
        nn.Module.__init__(self)
        if window_size < 2:
            raise ValueError("RNN-Attention window_size must be at least 2.")
        self.window_size = window_size
        self.output_size = 1
        self.output_offset = window_size // 2
        self._config = {"window_size": window_size}
        self.normalization = None
        self.conv1d = nn.Conv1d(1, 16, kernel_size=4, stride=1, padding=2)
        self.lstm1 = nn.LSTM(16, 128, num_layers=1, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(256, 256, num_layers=1, batch_first=True, bidirectional=True)
        self.attention = AttentionLayer(units=128)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 1)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, parameter in module.named_parameters():
                    if "weight" in name:
                        nn.init.xavier_uniform_(parameter)
                    elif "bias" in name:
                        nn.init.zeros_(parameter)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1).permute(0, 2, 1)
        x = self.conv1d(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.attention(x)
        x = torch.tanh(self.fc1(x))
        return self.fc2(x).reshape(-1)
