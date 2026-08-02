"""
BERT-inspired NILM model matching nilmtk-contrib architecture.

Official source: https://github.com/nilmtk/nilmtk-contrib/blob/14efd545e09d836e02159b23444350af92c6ee70/nilmtk_contrib/torch/bert.py
Local adaptation: repository registry and plugin interfaces.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model_pipeline.model_registry import register_model
from model_pipeline.models.seq2seq.seq2seq import Seq2SeqCNN


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
class BERTNILM(Seq2SeqCNN):
    display_name = "BERT"
    model_family = "transformer"
    target_type = "sequence"
    default_output_size = None
    default_window_size = 99
    default_num_epochs = 10
    official_batch_size = 512
    official_mains_mean = 1800.0
    official_mains_std = 600.0

    def __init__(self, *, window_size: int = 99, embed_dim: int = 32, num_heads: int = 2, ff_dim: int = 32):
        nn.Module.__init__(self)
        if (window_size, embed_dim, num_heads, ff_dim) != (99, 32, 2, 32):
            raise ValueError("BERT uses its official defaults (99, 32, 2, 32).")
        self.window_size = window_size
        self.output_size = window_size
        self.output_offset = 0
        self._config = {"window_size": window_size, "embed_dim": embed_dim, "num_heads": num_heads, "ff_dim": ff_dim}
        self.normalization = None
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
