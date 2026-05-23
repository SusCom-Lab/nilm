"""
Sequence-to-sequence CNN baseline for NILM.

This model is implemented for this project as a compact PyTorch baseline.
The closest confirmed open-source NILM baseline collection is:
https://github.com/nilmtk/nilmtk-contrib

Paper references:
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


@register_model("seq2seq", aliases=("Seq2Seq",), display_name="Seq2Seq")
class Seq2SeqCNN(TorchNILMModel):
    display_name = "Seq2Seq"
    model_family = "seq2seq"
    target_type = "sequence"
    default_output_size = None

    def __init__(self, *, window_size: int = 599, hidden_channels: int = 64, **kwargs):
        super().__init__(window_size=window_size, hidden_channels=hidden_channels, **kwargs)
        self.network = nn.Sequential(
            nn.Conv1d(1, hidden_channels, kernel_size=9, padding="same"),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=7, padding="same"),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=5, padding="same"),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.network(x)
        return x.squeeze(1)
