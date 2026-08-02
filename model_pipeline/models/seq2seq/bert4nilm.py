"""
BERT4NILM architecture adapted to the local NILM model registry.

Reference implementation:
https://github.com/Yueeeeeeee/BERT4NILM

Reference revision:
0e6b652b56e26c93c5396e391a4c100304974b18

The original model predicts an appliance-power sequence from an aggregate
window. Local adaptations only replace dataset I/O, household splitting,
checkpoint persistence, and the prediction return contract.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_pipeline.contracts import FitResult
from model_pipeline.model_registry import register_model
from model_pipeline.models.base_model import TorchNILMModel


class GELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class PositionalEmbedding(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        return self.pe.weight.unsqueeze(0).repeat(batch_size, 1, 1)


class BERT4NILMLayerNorm(nn.Module):
    def __init__(self, features: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.weight * (x - mean) / (std + self.eps) + self.bias


class Attention(nn.Module):
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
        dropout: nn.Module | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, heads: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        if d_model % heads != 0:
            raise ValueError("d_model must be divisible by heads.")
        self.d_k = d_model // heads
        self.heads = heads
        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = query.size(0)
        query, key, value = [
            layer(x).view(batch_size, -1, self.heads, self.d_k).transpose(1, 2)
            for layer, x in zip(self.linear_layers, (query, key, value))
        ]
        x, _attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.heads * self.d_k)
        return self.output_linear(x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.activation = GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_2(self.activation(self.w_1(x)))


class SublayerConnection(nn.Module):
    def __init__(self, size: int, dropout: float):
        super().__init__()
        self.layer_norm = BERT4NILMLayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer) -> torch.Tensor:
        return self.layer_norm(x + self.dropout(sublayer(x)))


class TransformerBlock(nn.Module):
    def __init__(self, hidden: int, attn_heads: int, feed_forward_hidden: int, dropout: float):
        super().__init__()
        self.attention = MultiHeadedAttention(heads=attn_heads, d_model=hidden, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.input_sublayer(x, lambda _x: self.attention(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)


@register_model(
    "bert4nilm",
    aliases=("BERT4NILM", "bert_4_nilm"),
    display_name="BERT4NILM",
)
class BERT4NILM(TorchNILMModel):
    display_name = "BERT4NILM"
    model_family = "transformer"
    target_type = "sequence"
    default_window_size = 480
    default_output_size = None
    requires_status_targets = True

    def __init__(
        self,
        *,
        window_size: int = 480,
        hidden_dim: int = 256,
        num_heads: int = 2,
        num_layers: int = 2,
        dropout: float = 0.1,
        appliance_output_size: int = 1,
        **kwargs,
    ):
        super().__init__(
            window_size=window_size,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            appliance_output_size=appliance_output_size,
            **kwargs,
        )
        self.hidden = int(hidden_dim)
        self.heads = int(num_heads)
        self.n_layers = int(num_layers)
        self.dropout_rate = float(dropout)
        self.appliance_output_size = int(appliance_output_size)
        self.latent_len = self.window_size // 2

        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=self.hidden,
            kernel_size=5,
            stride=1,
            padding=2,
            padding_mode="replicate",
        )
        self.pool = nn.LPPool1d(norm_type=2, kernel_size=2, stride=2)
        self.position = PositionalEmbedding(max_len=self.latent_len, d_model=self.hidden)
        self.layer_norm = BERT4NILMLayerNorm(self.hidden)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(self.hidden, self.heads, self.hidden * 4, self.dropout_rate)
                for _ in range(self.n_layers)
            ]
        )
        self.deconv = nn.ConvTranspose1d(
            in_channels=self.hidden,
            out_channels=self.hidden,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.linear1 = nn.Linear(self.hidden, 128)
        self.linear2 = nn.Linear(128, self.appliance_output_size)
        self.truncated_normal_init()

    def truncated_normal_init(self, mean: float = 0.0, std: float = 0.02, lower: float = -0.04, upper: float = 0.04):
        for name, parameter in self.named_parameters():
            if "layer_norm" in name:
                continue
            with torch.no_grad():
                l_value = (1.0 + math.erf(((lower - mean) / std) / math.sqrt(2.0))) / 2.0
                u_value = (1.0 + math.erf(((upper - mean) / std) / math.sqrt(2.0))) / 2.0
                parameter.uniform_(2 * l_value - 1, 2 * u_value - 1)
                parameter.erfinv_()
                parameter.mul_(std * math.sqrt(2.0))
                parameter.add_(mean)

    def _match_window_length(self, x: torch.Tensor) -> torch.Tensor:
        current_len = x.size(1)
        if current_len == self.window_size:
            return x
        if current_len > self.window_size:
            return x[:, : self.window_size, :]
        pad_len = self.window_size - current_len
        return F.pad(x, (0, 0, 0, pad_len))

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        x_token = self.pool(self.conv(sequence.unsqueeze(1))).permute(0, 2, 1)
        embedding = x_token + self.position(sequence)
        x = self.dropout(self.layer_norm(embedding))
        for transformer in self.transformer_blocks:
            x = transformer(x, mask=None)
        x = self.deconv(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self._match_window_length(x)
        x = torch.tanh(self.linear1(x))
        x = self.linear2(x)
        if self.appliance_output_size == 1:
            return x.squeeze(-1)
        return x

    def _split_supervised_targets(self, targets: torch.Tensor):
        if targets.ndim != 3 or targets.size(-1) != 2:
            raise ValueError(
                "BERT4NILM requires sequence targets containing power and status."
            )
        return targets[..., 0].float(), targets[..., 1].float()

    def _masked_batch_loss(self, inputs, targets, stats):
        power_targets, status_targets = self._split_supervised_targets(targets)
        mask_probability = float(self._config.get("mask_probability", 0.25))
        if not 0.0 < mask_probability <= 1.0:
            raise ValueError("BERT4NILM mask_probability must be in (0, 1].")

        selected = torch.rand_like(inputs) < mask_probability
        if not bool(selected.any()):
            selected.reshape(-1)[torch.randint(selected.numel(), (1,), device=inputs.device)] = True
        corruption = torch.rand_like(inputs)
        masked_inputs = inputs.clone()
        masked_inputs[selected & (corruption < 0.8)] = -1.0
        random_mask = selected & (corruption >= 0.8) & (corruption < 0.9)
        masked_inputs[random_mask] = torch.randn_like(masked_inputs[random_mask])

        outputs = self(masked_inputs)
        selected_outputs = outputs[selected].reshape(-1, 1)
        selected_power = power_targets[selected].reshape(-1, 1)
        selected_status = status_targets[selected].reshape(-1, 1)

        temperature = float(self._config.get("temperature", 0.1))
        kl_loss = F.kl_div(
            F.log_softmax(selected_outputs / temperature, dim=-1),
            F.softmax(selected_power / temperature, dim=-1),
            reduction="batchmean",
        )
        mse_loss = F.mse_loss(selected_outputs, selected_power)

        target_mean = float(stats["appliance_mean"])
        target_std = float(stats["appliance_std"])
        threshold = float(self._config.get("on_power_threshold", 15.0))
        raw_outputs = selected_outputs * target_std + target_mean
        predicted_status = (raw_outputs >= threshold).to(selected_outputs.dtype)
        margin_loss = F.soft_margin_loss(
            predicted_status * 2.0 - 1.0,
            selected_status * 2.0 - 1.0,
        )
        total_loss = kl_loss + mse_loss + margin_loss

        on_mask = (selected_status == 1) | (selected_status != predicted_status)
        if bool(on_mask.any()):
            on_loss = F.l1_loss(
                selected_outputs[on_mask],
                selected_power[on_mask],
                reduction="sum",
            )
            total_loss = total_loss + float(self._config.get("c0", 1.0)) * (
                on_loss / selected.numel()
            )
        return total_loss

    def _validation_mse(self, loader, device):
        if loader is None:
            return None
        self.eval()
        total = 0.0
        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(device)
                power_targets, _ = self._split_supervised_targets(targets.to(device))
                total += float(F.mse_loss(self(inputs), power_targets).item())
        return total / max(1, len(loader))

    def fit(self, train_data, validation_data, context) -> FitResult:
        """Run the official mask/corruption and composite-loss training recipe."""

        train_loader = train_data.loader
        validation_loader = validation_data.loader
        stats = train_data.normalisation_stats
        if train_loader is None or stats is None:
            raise ValueError("BERT4NILM requires a training loader and training statistics.")

        self.to(context.device)
        no_decay = ("bias", "layer_norm")
        named_parameters = list(self.named_parameters())
        parameter_groups = [
            {
                "params": [
                    parameter
                    for name, parameter in named_parameters
                    if not any(token in name for token in no_decay)
                ],
                "weight_decay": float(self._config.get("weight_decay", 0.0)),
            },
            {
                "params": [
                    parameter
                    for name, parameter in named_parameters
                    if any(token in name for token in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.Adam(
            parameter_groups,
            lr=float(self._config.get("learning_rate", 1e-4)),
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
                loss = self._masked_batch_loss(inputs, targets, stats)
                loss.backward()
                optimizer.step()
                total += float(loss.item())

            train_loss = total / max(1, len(train_loader))
            validation_loss = self._validation_mse(validation_loader, context.device)
            if validation_loss is None:
                validation_loss = train_loss
            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_mse": validation_loss,
                "selection_loss": validation_loss,
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
