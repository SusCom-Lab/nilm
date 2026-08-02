"""
Denoising autoencoder baseline for NILM.

Official source: https://github.com/nilmtk/nilmtk-contrib/blob/14efd545e09d836e02159b23444350af92c6ee70/nilmtk_contrib/torch/dae.py
Local adaptation: repository registry and plugin interfaces.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model_pipeline.model_registry import register_model
from model_pipeline.models.seq2seq.seq2seq import Seq2SeqCNN


@register_model("dae", aliases=("DAE", "Denoising Auto Encoder"), display_name="DAE")
class DenoisingAutoEncoder(Seq2SeqCNN):
    display_name = "DAE"
    model_family = "dae"
    target_type = "sequence"
    default_output_size = None
    default_window_size = 99
    default_num_epochs = 10
    official_batch_size = 512
    official_mains_mean = 1000.0
    official_mains_std = 600.0

    def __init__(self, *, window_size: int = 99):
        nn.Module.__init__(self)
        if window_size != 99:
            raise ValueError("DAE uses the official window_size=99.")
        self.window_size = window_size
        self.output_size = window_size
        self.output_offset = 0
        self._config = {"window_size": window_size}
        self.normalization = None
        flattened = self.window_size * 8
        self.conv1 = nn.Conv1d(1, 8, kernel_size=4, padding="same")
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(flattened, flattened)
        self.fc2 = nn.Linear(flattened, 128)
        self.fc3 = nn.Linear(128, flattened)
        self.conv2 = nn.Conv1d(8, 1, kernel_size=4, padding="same")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1).permute(0, 2, 1)
        x = torch.relu(self.conv1(x))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = x.view(-1, 8, self.window_size)
        x = self.conv2(x)
        x = x.permute(0, 2, 1)
        return x.squeeze(-1)
