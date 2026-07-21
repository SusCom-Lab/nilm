"""
BERT-inspired NILM model matching nilmtk-contrib architecture.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model_pipeline.model_registry import register_model
from model_pipeline.models.base_model import TorchNILMModel


class Permute(nn.Module):
    def __init__(self, *dims: int):
        super().__init__()
        self.dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(*self.dims)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, rate: float = 0.1):
        super().__init__()
        self.att = nn.MultiheadAttention(embed_dim, num_heads, dropout=rate, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.layernorm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_output, _ = self.att(x, x, x)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, maxlen: int):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.randn(1, maxlen, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos_emb


class LPpool(nn.Module):
    def __init__(self, pool_size: int, stride: int | None = None, padding: int = 0):
        super().__init__()
        self.avgpool = nn.AvgPool1d(pool_size, stride=stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.pow(torch.abs(x), 2)
        x = self.avgpool(x)
        return torch.pow(x, 0.5)


@register_model("bert", aliases=("BERT",), display_name="BERT")
class BERTNILM(TorchNILMModel):
    display_name = "BERT"
    model_family = "transformer"
    target_type = "sequence"
    default_output_size = None

    def __init__(self, *, window_size: int = 99, embed_dim: int = 32, num_heads: int = 2, ff_dim: int = 32, **kwargs):
        super().__init__(
            window_size=window_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            **kwargs,
        )
        pooled_length = ((self.window_size - 2) // 2) + 1
        self.network = nn.Sequential(
            Permute(0, 2, 1),
            nn.Conv1d(1, embed_dim, 4, stride=1, padding="same"),
            LPpool(pool_size=2),
            Permute(0, 2, 1),
            PositionalEncoding(embed_dim, pooled_length),
            TransformerBlock(embed_dim, num_heads, ff_dim),
            nn.Flatten(),
            nn.Dropout(0.1),
            nn.Linear(pooled_length * embed_dim, self.window_size),
            nn.Dropout(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)
        return self.network(x)
