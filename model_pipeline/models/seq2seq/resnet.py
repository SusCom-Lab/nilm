"""
ResNet NILM model matching nilmtk-contrib architecture.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_pipeline.model_registry import register_model
from model_pipeline.models.base_model import TorchNILMModel


class IdentityBlock(nn.Module):
    def __init__(self, filters: tuple[int, int, int], kernel_size: int, input_channels: int | None = None):
        super().__init__()
        in_channels = input_channels if input_channels is not None else filters[0]
        self.conv1 = nn.Conv1d(in_channels, filters[0], kernel_size, stride=1, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(filters[0], filters[1], kernel_size, stride=1, padding=kernel_size // 2)
        self.conv3 = nn.Conv1d(filters[1], filters[2], kernel_size, stride=1, padding=kernel_size // 2)
        self.shortcut = (
            nn.Conv1d(in_channels, filters[2], kernel_size=1, stride=1, padding=0)
            if in_channels != filters[2]
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = self.conv3(out)
        if out.size() != identity.size():
            min_size = min(out.size(2), identity.size(2))
            out = out[:, :, :min_size]
            identity = identity[:, :, :min_size]
        return F.relu(out + identity)


class ConvolutionBlock(nn.Module):
    def __init__(self, filters: tuple[int, int, int], kernel_size: int, input_channels: int | None = None):
        super().__init__()
        in_channels = input_channels if input_channels is not None else filters[0]
        self.conv1 = nn.Conv1d(in_channels, filters[0], kernel_size, stride=1, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(filters[0], filters[1], kernel_size, stride=1, padding=kernel_size // 2)
        self.conv3 = nn.Conv1d(filters[1], filters[2], kernel_size, stride=1, padding=kernel_size // 2)
        self.conv4 = nn.Conv1d(in_channels, filters[2], kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = F.relu(self.conv4(x))
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        if out.size() != identity.size():
            min_size = min(out.size(2), identity.size(2))
            out = out[:, :, :min_size]
            identity = identity[:, :, :min_size]
        return F.relu(out + identity)


@register_model("resnet", aliases=("ResNet",), display_name="ResNet")
class ResNetNILM(TorchNILMModel):
    display_name = "ResNet"
    model_family = "cnn"
    target_type = "sequence"
    default_output_size = None

    def __init__(self, *, window_size: int = 299, num_filters: int = 30, hidden_dim: int = 1024, **kwargs):
        super().__init__(window_size=window_size, num_filters=num_filters, hidden_dim=hidden_dim, **kwargs)
        self.zero_pad = nn.ZeroPad1d(3)
        self.conv1 = nn.Conv1d(1, num_filters, kernel_size=48, stride=2, padding=0)
        self.bn1 = nn.BatchNorm1d(num_filters)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=0)
        self.conv_block = ConvolutionBlock((num_filters, num_filters, num_filters), 24, input_channels=num_filters)
        self.identity_block1 = IdentityBlock((num_filters, num_filters, num_filters), 12, input_channels=num_filters)
        self.identity_block2 = IdentityBlock((num_filters, num_filters, num_filters), 6, input_channels=num_filters)
        self.fc_input_size = self._calculate_fc_input_size()
        self.fc1 = nn.Linear(self.fc_input_size, hidden_dim)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_dim, self.window_size)

    def _forward_conv_layers(self, x: torch.Tensor) -> torch.Tensor:
        x = self.zero_pad(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.conv_block(x)
        x = self.identity_block1(x)
        x = self.identity_block2(x)
        return x

    def _calculate_fc_input_size(self) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.window_size)
            x = self._forward_conv_layers(dummy)
            return x.view(x.size(0), -1).size(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self._forward_conv_layers(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)
