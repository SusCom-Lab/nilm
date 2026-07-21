"""
Sequence-to-sequence CNN baseline for NILM.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model_pipeline.model_registry import register_model
from model_pipeline.models.base_model import TorchNILMModel


@register_model("seq2seq", aliases=("Seq2Seq",), display_name="Seq2Seq")
class Seq2SeqCNN(TorchNILMModel):
    display_name = "Seq2Seq"
    model_family = "seq2seq"
    target_type = "sequence"
    default_output_size = None

    def __init__(self, *, window_size: int = 599, hidden_dim: int = 1024, **kwargs):
        super().__init__(window_size=window_size, hidden_dim=hidden_dim, **kwargs)
        self.hidden_dim = int(hidden_dim)
        length = self.window_size
        length = (length - 10) // 2 + 1
        length = (length - 8) // 2 + 1
        length = length - 6 + 1
        length = length - 5 + 1
        length = length - 5 + 1
        flattened = 50 * length

        self.conv1 = nn.Conv1d(1, 30, 10, stride=2)
        self.conv2 = nn.Conv1d(30, 30, 8, stride=2)
        self.conv3 = nn.Conv1d(30, 40, 6, stride=1)
        self.conv4 = nn.Conv1d(40, 50, 5, stride=1)
        self.dropout1 = nn.Dropout(0.2)
        self.conv5 = nn.Conv1d(50, 50, 5, stride=1)
        self.dropout2 = nn.Dropout(0.2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(flattened, self.hidden_dim)
        self.dropout3 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(self.hidden_dim, self.output_size)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1).permute(0, 2, 1)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = self.dropout1(x)
        x = torch.relu(self.conv5(x))
        x = self.dropout2(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        return self.dropout3(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.encode(x))
