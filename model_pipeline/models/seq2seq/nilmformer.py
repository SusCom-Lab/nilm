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

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model_pipeline.api import FitResult, InferenceContext, PredictionOutput, TrainingContext
from model_pipeline.model_registry import register_model
from model_pipeline.models.seq2seq.seq2seq import Seq2SeqCNN


def _calendar_features(timestamps) -> np.ndarray:
    values = pd.to_datetime(pd.Series(timestamps), errors="coerce")
    if values.isna().any():
        raise ValueError("NILMFormer requires valid timestamps.")
    periodic = (
        (values.dt.minute.to_numpy(dtype=np.float32), 60.0),
        (values.dt.hour.to_numpy(dtype=np.float32), 24.0),
        (values.dt.dayofweek.to_numpy(dtype=np.float32), 7.0),
        (values.dt.month.to_numpy(dtype=np.float32), 12.0),
    )
    channels = []
    for raw, period in periodic:
        phase = 2.0 * np.pi * raw / period
        channels.extend((np.sin(phase), np.cos(phase)))
    return np.stack(channels).astype(np.float32, copy=False)


class NILMFormerDataset(Dataset):
    def __init__(self, partition, window_size, stats):
        self.inputs = []
        self.targets = []
        half = window_size // 2
        for series in partition.series:
            if len(series.appliances) != 1:
                raise ValueError("NILMFormer is configured for one appliance.")
            segments = series.segment_ids
            if segments is None:
                segments = np.zeros(len(series.aggregate), dtype=np.int64)
            start = 0
            while start < len(series.aggregate):
                end = start + 1
                while end < len(series.aggregate) and segments[end] == segments[start]:
                    end += 1
                mains = np.pad(series.aggregate[start:end], (half, half))
                calendar = np.pad(
                    _calendar_features(series.timestamps[start:end]),
                    ((0, 0), (half, half)), mode="edge",
                )
                target = np.pad(series.appliance_power[start:end, 0], (half, half))
                for index in range(end - start):
                    power_window = (
                        mains[index:index + window_size] - stats["mains_mean"]
                    ) / stats["mains_std"]
                    self.inputs.append(torch.as_tensor(
                        np.concatenate((power_window[None, :], calendar[:, index:index + window_size])),
                        dtype=torch.float32,
                    ))
                    self.targets.append(torch.as_tensor(
                        (target[index:index + window_size] - stats["appliance_mean"])
                        / stats["appliance_std"], dtype=torch.float32
                    ))
                start = end

    def __len__(self): return len(self.inputs)
    def __getitem__(self, index): return self.inputs[index], self.targets[index]


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
class NILMFormer(Seq2SeqCNN):
    """Official NILMFormer core with real calendar features supplied by the dataset."""

    display_name = "NILMFormer"
    model_family = "transformer"
    target_type = "sequence"
    default_window_size = 256
    default_output_size = None
    requires_temporal_features = True
    default_num_epochs = 50
    official_batch_size = 128
    official_mains_mean = 0.0
    official_mains_std = 1.0

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
    ) -> None:
        dilations = list(dilations or [1, 2, 4, 8])
        if c_embedding != 8:
            raise ValueError("Real calendar encoding supplies exactly 8 channels.")
        if d_model % 4 != 0:
            raise ValueError("d_model must be divisible by 4.")
        nn.Module.__init__(self)
        if window_size < 2 or kernel_size < 1 or kernel_size_head < 1:
            raise ValueError("NILMFormer window and kernel sizes must be positive.")
        if n_encoder_layers < 1 or d_model < 1 or pffn_ratio < 1 or n_head < 1:
            raise ValueError("NILMFormer layer dimensions must be positive.")
        if d_model % n_head != 0 or not 0.0 <= dp_rate < 1.0 or norm_eps <= 0:
            raise ValueError("Invalid NILMFormer attention, dropout, or normalization settings.")
        self.window_size = window_size
        self.output_size = window_size
        self.output_offset = 0
        self._config = {
            "window_size": window_size, "c_embedding": c_embedding,
            "kernel_size": kernel_size, "kernel_size_head": kernel_size_head,
            "dilations": dilations, "conv_bias": conv_bias,
            "n_encoder_layers": n_encoder_layers, "d_model": d_model,
            "dp_rate": dp_rate, "pffn_ratio": pffn_ratio,
            "n_head": n_head, "norm_eps": norm_eps,
            "learning_rate": 1e-4, "weight_decay": 0.01,
            "warmup_fraction": 0.1, "gradient_clip_norm": 1.0,
        }
        self.normalization = None
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

    def fit(self, train_data, validation_data, context: TrainingContext) -> FitResult:
        """Run NILMFormer's AdamW, warmup/cosine, and clipping recipe."""

        del validation_data
        self._set_normalization(train_data)
        train_loader = DataLoader(
            NILMFormerDataset(train_data, self.window_size, self.normalization),
            batch_size=self.official_batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(context.seed),
        )

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
                targets = targets.to(context.device)
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
            validation = context.validate_candidate(epoch=epoch, model=self)
            record = {
                "epoch": epoch,
                "training_loss": train_loss,
                "validation_mse": validation.mse,
                "learning_rate": scheduler.get_last_lr()[0],
            }
            history.append(record)
            if validation.improved:
                best_epoch = epoch
                best_score = validation.mse
                checkpoint_path = validation.checkpoint_path
            if validation.should_stop:
                break

        return FitResult(
            best_epoch=best_epoch,
            history=history,
            best_validation_mse=best_score,
            checkpoint_path=checkpoint_path,
        )

    def _predict_segment(self, timestamps, aggregate, context):
        stats = self.normalization
        half = self.window_size // 2
        mains = np.pad(aggregate, (half, half))
        calendar = np.pad(
            _calendar_features(timestamps), ((0, 0), (half, half)), mode="edge"
        )
        windows = []
        for index in range(len(aggregate)):
            power_window = (
                mains[index:index + self.window_size] - stats["mains_mean"]
            ) / stats["mains_std"]
            windows.append(np.concatenate(
                (power_window[None, :], calendar[:, index:index + self.window_size])
            ))
        predicted = []
        with torch.inference_mode():
            for offset in range(0, len(windows), context.batch_size):
                inputs = torch.as_tensor(
                    np.asarray(windows[offset:offset + context.batch_size]),
                    dtype=torch.float32, device=context.device,
                )
                predicted.append(self(inputs).cpu().numpy())
        predicted = np.concatenate(predicted)
        predicted = predicted * stats["appliance_std"] + stats["appliance_mean"]
        sums = np.zeros(len(mains), dtype=np.float64)
        counts = np.zeros(len(mains), dtype=np.float64)
        for index, window in enumerate(predicted):
            sums[index:index + self.window_size] += window
            counts[index:index + self.window_size] += 1
        return (sums[half:half + len(aggregate)] / counts[half:half + len(aggregate)]).astype(np.float32)

    def predict(self, inference_data, context: InferenceContext):
        if self.normalization is None:
            raise RuntimeError("NILMFormer must be fitted or loaded before prediction.")
        self.to(context.device)
        self.eval()
        timestamp_parts, household_parts, power_parts = [], [], []
        for series in inference_data.series:
            segments = series.segment_ids
            if segments is None:
                segments = np.zeros(len(series.aggregate), dtype=np.int64)
            output = np.empty(len(series.aggregate), dtype=np.float32)
            start = 0
            while start < len(series.aggregate):
                end = start + 1
                while end < len(series.aggregate) and segments[end] == segments[start]: end += 1
                output[start:end] = self._predict_segment(
                    series.timestamps[start:end], series.aggregate[start:end], context
                )
                start = end
            timestamp_parts.append(series.timestamps)
            household_parts.append(np.repeat(series.household_id, len(series.timestamps)))
            power_parts.append(output[:, None])
        return PredictionOutput(
            np.concatenate(timestamp_parts), np.concatenate(power_parts),
            tuple(self.normalization["appliances"]), np.concatenate(household_parts)
        )
