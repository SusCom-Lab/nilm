"""
Denoising autoencoder baseline for NILM.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model_pipeline.model_registry import register_model
from model_pipeline.models.base_model import TorchNILMModel


@register_model("dae", aliases=("DAE", "Denoising Auto Encoder"), display_name="DAE")
class DenoisingAutoEncoder(TorchNILMModel):
    display_name = "DAE"
    model_family = "dae"
    target_type = "sequence"
    default_output_size = None

    def __init__(self, *, window_size: int = 599, **kwargs):
        super().__init__(window_size=window_size, **kwargs)
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
