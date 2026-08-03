"""
Denoising autoencoder baseline for NILM.

Official source: https://github.com/nilmtk/nilmtk-contrib/blob/14efd545e09d836e02159b23444350af92c6ee70/nilmtk_contrib/torch/dae.py
Local adaptation: repository registry and plugin interfaces.
"""

from __future__ import annotations

import numpy as np
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
    official_optimizer_epsilon = 1e-8

    def __init__(self, *, window_size: int = 99):
        nn.Module.__init__(self)
        if window_size < 1:
            raise ValueError("DAE window_size must be positive.")
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

    def _predict_segment(self, aggregate, context):
        """Use the official non-overlapping DAE test windows."""

        stats = self.normalization
        padding = (-len(aggregate)) % self.window_size
        values = np.pad(aggregate, (0, padding))
        windows = values.reshape(-1, self.window_size)
        windows = (windows - stats["mains_mean"]) / stats["mains_std"]
        predicted = []
        with torch.inference_mode():
            for offset in range(0, len(windows), context.batch_size):
                inputs = torch.as_tensor(
                    windows[offset : offset + context.batch_size],
                    dtype=torch.float32,
                    device=context.device,
                )
                predicted.append(self(inputs).cpu().numpy())
        output = np.concatenate(predicted, axis=0).reshape(-1)
        output = output * stats["appliance_std"] + stats["appliance_mean"]
        return output[: len(aggregate)].astype(np.float32)
