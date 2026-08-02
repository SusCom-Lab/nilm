"""NILMFormer adapted to the local sequence-model interface.

The network architecture is derived from the official Apache-2.0 implementation:
https://github.com/adrienpetralia/NILMFormer

Training recipe reference:
https://github.com/nilmtk/nilmtk-contrib/blob/14efd545e09d836e02159b23444350af92c6ee70/nilmtk_contrib/torch/nilmformer.py

Copyright © 2025 EDF
Local integration changes: registry metadata, input validation, and removal of the
singleton output channel so the model returns ``[batch, sequence_length]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_pipeline.contracts import FitResult
from model_pipeline.model_registry import register_model
from model_pipeline.models.base_model import TorchNILMModel


@dataclass
class NILMFormerConfig:
    c_in: int = 1
    c_embedding: int = 8
    c_out: int = 1
    kernel_size: int = 3
    kernel_size_head: int = 3
    dilations: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    conv_bias: bool = True
    n_encoder_layers: int = 3
    d_model: int = 96
    dp_rate: float = 0.2
    pffn_ratio: int = 4
    n_head: int = 8
    norm_eps: float = 1e-5


class ResUnit(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_out: int,
        k: int = 8,
        dilation: int = 1,
        stride: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(
                c_in,
                c_out,
                kernel_size=k,
                dilation=dilation,
                stride=stride,
                bias=bias,
                padding="same",
            ),
            nn.GELU(),
            nn.BatchNorm1d(c_out),
        )
        self.match_residual = c_in > 1 and c_in != c_out
        if self.match_residual:
            self.conv = nn.Conv1d(c_in, c_out, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.conv(x) if self.match_residual else x
        return residual + self.layers(x)


class DilatedBlock(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_out: int,
        kernel_size: int,
        dilation_list: list[int],
        bias: bool,
    ) -> None:
        super().__init__()
        layers = []
        for index, dilation in enumerate(dilation_list):
            layer_c_in = c_in if index == 0 else c_out
            layers.append(
                ResUnit(
                    layer_c_in,
                    c_out,
                    k=kernel_size,
                    dilation=dilation,
                    bias=bias,
                )
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class DiagonallyMaskedSelfAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, head_dim: int, dropout: float) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)
        self.wq = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wv = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, sequence_length, _ = x.shape
        query = self.wq(x).view(batch, sequence_length, self.n_heads, self.head_dim)
        key = self.wk(x).view(batch, sequence_length, self.n_heads, self.head_dim)
        value = self.wv(x).view(batch, sequence_length, self.n_heads, self.head_dim)

        scores = torch.einsum("blhe,bshe->bhls", query, key)
        diagonal_mask = torch.eye(
            sequence_length,
            dtype=torch.bool,
            device=x.device,
        ).view(1, 1, sequence_length, sequence_length)
        scores = scores.masked_fill(diagonal_mask, torch.finfo(scores.dtype).min)
        attention = self.attn_dropout(
            torch.softmax(scores * (self.head_dim**-0.5), dim=-1)
        )
        output = torch.einsum("bhls,bshd->blhd", attention, value)
        return self.out_dropout(self.wo(output.reshape(batch, sequence_length, -1)))


class PositionWiseFeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.layer1 = nn.Linear(dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer2(self.dropout(F.gelu(self.layer1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, config: NILMFormerConfig) -> None:
        super().__init__()
        if config.d_model % config.n_head != 0:
            raise ValueError("d_model must be divisible by n_head.")
        self.attention_layer = DiagonallyMaskedSelfAttention(
            config.d_model,
            config.n_head,
            config.d_model // config.n_head,
            config.dp_rate,
        )
        self.norm1 = nn.LayerNorm(config.d_model, eps=config.norm_eps)
        self.norm2 = nn.LayerNorm(config.d_model, eps=config.norm_eps)
        self.dropout = nn.Dropout(config.dp_rate)
        self.pffn = PositionWiseFeedForward(
            config.d_model,
            config.d_model * config.pffn_ratio,
            config.dp_rate,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x)
        x = x + self.attention_layer(x)
        x = self.norm2(x)
        return x + self.dropout(self.pffn(x))


@register_model(
    "nilmformer",
    aliases=("NILMFormer", "nilm_former"),
    display_name="NILMFormer",
)
class NILMFormer(TorchNILMModel):
    """Official NILMFormer core with real calendar features supplied by the dataset."""

    display_name = "NILMFormer"
    model_family = "transformer"
    target_type = "sequence"
    default_window_size = 256
    default_output_size = None
    requires_temporal_features = True

    def __init__(
        self,
        *,
        window_size: int = 256,
        c_embedding: int = 8,
        kernel_size: int = 3,
        kernel_size_head: int = 3,
        dilations: list[int] | None = None,
        conv_bias: bool = True,
        n_encoder_layers: int = 3,
        d_model: int = 96,
        dp_rate: float = 0.2,
        pffn_ratio: int = 4,
        n_head: int = 8,
        norm_eps: float = 1e-5,
        **kwargs,
    ) -> None:
        dilations = list(dilations or [1, 2, 4, 8])
        if c_embedding != 8:
            raise ValueError("Real calendar encoding supplies exactly 8 channels.")
        if d_model % 4 != 0:
            raise ValueError("d_model must be divisible by 4.")
        super().__init__(
            window_size=window_size,
            c_embedding=c_embedding,
            kernel_size=kernel_size,
            kernel_size_head=kernel_size_head,
            dilations=dilations,
            conv_bias=conv_bias,
            n_encoder_layers=n_encoder_layers,
            d_model=d_model,
            dp_rate=dp_rate,
            pffn_ratio=pffn_ratio,
            n_head=n_head,
            norm_eps=norm_eps,
            **kwargs,
        )
        config = NILMFormerConfig(
            c_embedding=c_embedding,
            kernel_size=kernel_size,
            kernel_size_head=kernel_size_head,
            dilations=dilations,
            conv_bias=conv_bias,
            n_encoder_layers=n_encoder_layers,
            d_model=d_model,
            dp_rate=dp_rate,
            pffn_ratio=pffn_ratio,
            n_head=n_head,
            norm_eps=norm_eps,
        )
        feature_dim = 3 * config.d_model // 4
        self.EmbedBlock = DilatedBlock(
            config.c_in,
            feature_dim,
            config.kernel_size,
            config.dilations,
            config.conv_bias,
        )
        self.ProjEmbedding = nn.Conv1d(
            config.c_embedding,
            config.d_model // 4,
            kernel_size=1,
        )
        self.ProjStats1 = nn.Linear(2, config.d_model)
        self.ProjStats2 = nn.Linear(config.d_model, 2)
        encoder_layers = [EncoderLayer(config) for _ in range(config.n_encoder_layers)]
        encoder_layers.append(nn.LayerNorm(config.d_model))
        self.EncoderBlock = nn.Sequential(*encoder_layers)
        self.DownstreamTaskHead = nn.Conv1d(
            config.d_model,
            config.c_out,
            kernel_size=config.kernel_size_head,
            padding=config.kernel_size_head // 2,
            padding_mode="replicate",
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3 or inputs.size(1) != 9:
            raise ValueError(
                "NILMFormer expects [batch, 9, sequence_length]: one aggregate "
                "channel followed by eight real calendar channels."
            )
        if inputs.size(-1) != self.window_size:
            raise ValueError(
                f"Expected sequence length {self.window_size}, got {inputs.size(-1)}."
            )

        aggregate = inputs[:, :1, :]
        calendar_encoding = inputs[:, 1:, :]
        instance_mean = aggregate.mean(dim=-1, keepdim=True).detach()
        instance_std = torch.sqrt(
            aggregate.var(dim=-1, keepdim=True, unbiased=False) + 1e-6
        ).detach()
        aggregate = (aggregate - instance_mean) / instance_std

        aggregate_embedding = self.EmbedBlock(aggregate)
        calendar_embedding = self.ProjEmbedding(calendar_encoding)
        tokens = torch.cat((aggregate_embedding, calendar_embedding), dim=1).permute(0, 2, 1)
        stats_token = self.ProjStats1(
            torch.cat((instance_mean, instance_std), dim=1).permute(0, 2, 1)
        )
        tokens = self.EncoderBlock(torch.cat((tokens, stats_token), dim=1))[:, :-1, :]
        output = self.DownstreamTaskHead(tokens.permute(0, 2, 1))

        output_stats = self.ProjStats2(stats_token)
        output_mean = output_stats[:, :, 0].unsqueeze(-1)
        output_std = output_stats[:, :, 1].unsqueeze(-1)
        return (output * output_std + output_mean).squeeze(1)

    def _validation_mse(self, loader, device):
        if loader is None:
            return None
        self.eval()
        total = 0.0
        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(device)
                targets = self.prepare_targets(targets.to(device))
                total += float(F.mse_loss(self(inputs), targets).item())
        return total / max(1, len(loader))

    def fit(self, train_data, validation_data, context) -> FitResult:
        """Run NILMFormer's AdamW, warmup/cosine, and clipping recipe."""

        train_loader = train_data.loader
        validation_loader = validation_data.loader
        if train_loader is None or len(train_loader) == 0:
            raise ValueError("NILMFormer requires a non-empty training loader.")

        self.to(context.device)
        learning_rate = float(self._config.get("learning_rate", 1e-4))
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=learning_rate,
            weight_decay=float(self._config.get("weight_decay", 0.01)),
            betas=(0.9, 0.95),
        )
        total_steps = len(train_loader) * int(context.num_epochs)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            total_steps=total_steps,
            pct_start=float(self._config.get("warmup_fraction", 0.1)),
            anneal_strategy="cos",
        )
        history = []
        best_epoch = None
        best_score = None
        checkpoint_path = None

        for epoch in range(1, int(context.num_epochs) + 1):
            self.train()
            total = 0.0
            for inputs, targets in train_loader:
                inputs = inputs.to(context.device)
                targets = self.prepare_targets(targets.to(context.device))
                optimizer.zero_grad()
                loss = F.mse_loss(self(inputs), targets)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.parameters(),
                    max_norm=float(self._config.get("gradient_clip_norm", 1.0)),
                )
                optimizer.step()
                scheduler.step()
                total += float(loss.item())

            train_loss = total / max(1, len(train_loader))
            validation_loss = self._validation_mse(validation_loader, context.device)
            if validation_loss is None:
                validation_loss = train_loss
            record = {
                "epoch": epoch,
                "train_mse": train_loss,
                "validation_mse": validation_loss,
                "selection_loss": validation_loss,
                "learning_rate": scheduler.get_last_lr()[0],
                "checkpoint_eligible": True,
            }
            history.append(record)
            print(
                f"Epoch {epoch}/{context.num_epochs}, Train Loss: {train_loss}, "
                f"Val Loss: {validation_loss}, Selection Loss: {validation_loss}"
            )
            decision = context.checkpoint_callback(
                model=self,
                epoch=epoch,
                train_loss=train_loss,
                validation_loss=validation_loss,
                selection_loss=validation_loss,
                record=record,
            ) or {}
            if decision.get("saved"):
                best_epoch = epoch
                best_score = validation_loss
                checkpoint_path = decision.get("checkpoint_path")
            if decision.get("should_stop"):
                break

        return FitResult(
            best_epoch=best_epoch,
            best_score=best_score,
            history=history,
            checkpoint_path=checkpoint_path,
        )
