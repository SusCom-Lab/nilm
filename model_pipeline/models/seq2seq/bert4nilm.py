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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model_pipeline.api import (
    FitResult,
    InferenceContext,
    PredictionOutput,
    TrainingContext,
)
from model_pipeline.data_protocol import (
    get_appliance_status_rule,
    normalize_appliance_name,
)
from model_pipeline.model_registry import register_model
from model_pipeline.models.seq2seq.seq2seq import Seq2SeqCNN


class BERT4NILMDataset(Dataset):
    """Official forward windows with aligned power/status supervision."""

    def __init__(self, partition, window_size, stride, stats):
        self.inputs = []
        self.targets = []
        for series in partition.series:
            if len(series.appliances) != 1 or series.status is None:
                raise ValueError("BERT4NILM requires one appliance and public status labels.")
            segments = series.segment_ids
            if segments is None:
                segments = np.zeros(len(series.aggregate), dtype=np.int64)
            start = 0
            while start < len(series.aggregate):
                end = start + 1
                while end < len(series.aggregate) and segments[end] == segments[start]:
                    end += 1
                mains = _official_energy(series.aggregate[start:end], stats["aggregate_cutoff"])
                power = _official_energy(
                    series.appliance_power[start:end, 0], stats["power_cutoff"]
                )
                status = series.status[start:end, 0]
                for offset in _forward_window_starts(len(mains), window_size, stride):
                    valid = min(window_size, len(mains) - offset)
                    input_window = np.zeros(window_size, dtype=np.float32)
                    power_window = np.zeros(window_size, dtype=np.float32)
                    status_window = np.zeros(window_size, dtype=np.float32)
                    input_window[:valid] = (
                        mains[offset : offset + valid] - stats["mains_mean"]
                    ) / stats["mains_std"]
                    power_window[:valid] = power[offset : offset + valid]
                    status_window[:valid] = status[offset : offset + valid]
                    power_window /= stats["power_cutoff"]
                    self.inputs.append(torch.as_tensor(input_window, dtype=torch.float32))
                    self.targets.append(
                        torch.as_tensor(
                            np.stack((power_window, status_window), axis=-1),
                            dtype=torch.float32,
                        )
                    )
                start = end

    def __len__(self): return len(self.inputs)
    def __getitem__(self, index): return self.inputs[index], self.targets[index]


def _official_energy(values, cutoff):
    values = np.clip(np.asarray(values, dtype=np.float32), 0.0, float(cutoff)).copy()
    values[values < 5.0] = 0.0
    return values


def _forward_window_starts(length: int, window_size: int, stride: int) -> list[int]:
    if length <= window_size:
        return [0]
    count = int(np.ceil((length - window_size) / stride)) + 1
    return [index * stride for index in range(count)]


OFFICIAL_RECIPES = {
    "redd": {
        "stride": 120,
        "cutoff": {
            "fridge": 400.0,
            "washing_machine": 3500.0,
            "microwave": 1800.0,
            "dishwasher": 1200.0,
        },
        "c0": {
            "fridge": 1e-6,
            "washing_machine": 0.001,
            "microwave": 1.0,
            "dishwasher": 1.0,
        },
    },
    "ukdale": {
        "stride": 240,
        "cutoff": {
            "kettle": 3100.0,
            "fridge": 300.0,
            "washing_machine": 2500.0,
            "microwave": 3000.0,
            "dishwasher": 2500.0,
        },
        "c0": {
            "kettle": 1.0,
            "fridge": 1e-6,
            "washing_machine": 0.01,
            "microwave": 1.0,
            "dishwasher": 1.0,
        },
    },
}


def _dataset_key(value: str) -> str:
    compact = "".join(character for character in str(value).lower() if character.isalnum())
    if compact.startswith("redd"):
        return "redd"
    if compact in {"ukdale", "ukdalelowfrequency"}:
        return "ukdale"
    return compact


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
class BERT4NILM(Seq2SeqCNN):
    display_name = "BERT4NILM"
    model_family = "transformer"
    target_type = "sequence"
    default_window_size = 480
    default_output_size = None
    requires_status_targets = True
    default_num_epochs = 100
    official_batch_size = 128
    official_mains_mean = 0.0
    official_mains_std = 1.0

    def __init__(
        self,
        *,
        window_size: int = 480,
        hidden_dim: int = 256,
        num_heads: int = 2,
        num_layers: int = 2,
        dropout: float = 0.1,
        appliance_output_size: int = 1,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
        mask_probability: float = 0.25,
        temperature: float = 0.1,
        aggregate_cutoff: float = 6000.0,
        power_cutoff: float | None = None,
        status_threshold: float | None = None,
        train_stride: int | None = None,
        c0: float | None = None,
    ):
        nn.Module.__init__(self)
        if window_size < 2 or window_size % 2:
            raise ValueError("BERT4NILM window_size must be a positive even integer.")
        if hidden_dim < 1 or num_heads < 1 or hidden_dim % num_heads:
            raise ValueError("BERT4NILM hidden_dim must be divisible by num_heads.")
        if num_layers < 1 or not 0.0 <= dropout < 1.0 or appliance_output_size < 1:
            raise ValueError("BERT4NILM layer counts must be positive and dropout must be in [0, 1).")
        if learning_rate <= 0 or weight_decay < 0 or not 0 < mask_probability <= 1:
            raise ValueError("Invalid BERT4NILM optimiser or mask configuration.")
        if aggregate_cutoff <= 0 or (power_cutoff is not None and power_cutoff <= 0):
            raise ValueError("BERT4NILM cutoffs must be positive.")
        if train_stride is not None and train_stride < 1:
            raise ValueError("BERT4NILM train_stride must be positive.")
        self.window_size = window_size
        self.output_size = window_size
        self.output_offset = 0
        self._config = {
            "window_size": window_size, "hidden_dim": hidden_dim,
            "num_heads": num_heads, "num_layers": num_layers,
            "dropout": dropout, "appliance_output_size": appliance_output_size,
            "learning_rate": learning_rate, "weight_decay": weight_decay,
            "mask_probability": mask_probability, "temperature": temperature,
            "aggregate_cutoff": aggregate_cutoff,
            "power_cutoff": power_cutoff, "status_threshold": status_threshold,
            "train_stride": train_stride, "c0": c0,
        }
        self._recipe_overrides = {
            "power_cutoff": power_cutoff,
            "status_threshold": status_threshold,
            "train_stride": train_stride,
            "c0": c0,
        }
        self.normalization = None
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

    def _set_official_normalization(self, train_data, context: TrainingContext) -> None:
        appliance = normalize_appliance_name(train_data.series[0].appliances[0])
        dataset = _dataset_key(context.metadata.get("dataset", ""))
        recipe = OFFICIAL_RECIPES.get(dataset)
        public_rule = get_appliance_status_rule(appliance)
        cutoff = self._recipe_overrides["power_cutoff"]
        if cutoff is None and recipe is not None:
            cutoff = recipe["cutoff"].get(appliance)
        if cutoff is None and public_rule is not None:
            cutoff = public_rule["max_threshold"]
        if cutoff is None:
            raise ValueError(
                f"BERT4NILM has no official power cutoff for '{appliance}'; "
                "provide power_cutoff through the model CLI parameters."
            )
        threshold = self._recipe_overrides["status_threshold"]
        if threshold is None and public_rule is not None:
            threshold = public_rule["status_threshold"]
        if threshold is None:
            raise ValueError(f"No public status threshold exists for '{appliance}'.")
        stride = self._recipe_overrides["train_stride"]
        if stride is None:
            stride = recipe["stride"] if recipe is not None else 120
        c0 = self._recipe_overrides["c0"]
        if c0 is None and recipe is not None:
            c0 = recipe["c0"].get(appliance)
        if c0 is None:
            c0 = OFFICIAL_RECIPES["ukdale"]["c0"].get(appliance, 1.0)

        aggregate_cutoff = float(self._config["aggregate_cutoff"])
        aggregate = np.concatenate(
            [
                _official_energy(series.aggregate, aggregate_cutoff)
                for series in train_data.series
            ]
        )
        mains_std = float(np.std(aggregate))
        if not np.isfinite(mains_std) or mains_std <= 0:
            mains_std = 1.0
        self.normalization = {
            "mains_mean": float(np.mean(aggregate)),
            "mains_std": mains_std,
            "aggregate_cutoff": aggregate_cutoff,
            "power_cutoff": float(cutoff),
            "status_threshold": float(threshold),
            "train_stride": int(stride),
            "c0": float(c0),
            "appliances": train_data.series[0].appliances,
            "recipe_dataset": dataset or "custom",
        }
        self._config.update(
            power_cutoff=float(cutoff),
            status_threshold=float(threshold),
            train_stride=int(stride),
            c0=float(c0),
        )

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

        threshold = float(stats["status_threshold"])
        raw_outputs = selected_outputs * float(stats["power_cutoff"])
        raw_outputs = torch.clamp(raw_outputs, min=0.0, max=float(stats["power_cutoff"]))
        raw_outputs = torch.where(raw_outputs < 5.0, 0.0, raw_outputs)
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
            total_loss = total_loss + float(stats["c0"]) * (
                on_loss / selected.numel()
            )
        return total_loss

    def fit(self, train_data, validation_data, context: TrainingContext) -> FitResult:
        """Run the official mask/corruption and composite-loss training recipe."""

        del validation_data
        self._set_official_normalization(train_data, context)
        stats = self.normalization
        train_loader = DataLoader(
            BERT4NILMDataset(
                train_data,
                self.window_size,
                int(stats["train_stride"]),
                stats,
            ),
            batch_size=self.official_batch_size,
            shuffle=False,
        )

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
            validation = context.validate_candidate(epoch=epoch, model=self)
            record = {
                "epoch": epoch,
                "training_loss": train_loss,
                "validation_mse": validation.mse,
            }
            history.append(record)
            if validation.improved:
                best_epoch = epoch
                best_score = validation.mse
                checkpoint_path = validation.checkpoint_path
            if validation.should_stop:
                break

        return FitResult(
            history=history,
            best_epoch=best_epoch,
            best_validation_mse=best_score,
            checkpoint_path=checkpoint_path,
        )

    def _predict_segment(self, aggregate, context: InferenceContext) -> np.ndarray:
        stats = self.normalization
        mains = _official_energy(aggregate, stats["aggregate_cutoff"])
        starts = _forward_window_starts(len(mains), self.window_size, self.window_size)
        windows = np.zeros((len(starts), self.window_size), dtype=np.float32)
        valid_lengths = []
        for index, start in enumerate(starts):
            valid = min(self.window_size, len(mains) - start)
            valid_lengths.append(valid)
            windows[index, :valid] = (
                mains[start : start + valid] - stats["mains_mean"]
            ) / stats["mains_std"]

        predicted = []
        with torch.inference_mode():
            for offset in range(0, len(windows), context.batch_size):
                inputs = torch.as_tensor(
                    windows[offset : offset + context.batch_size],
                    dtype=torch.float32,
                    device=context.device,
                )
                predicted.append(self(inputs).cpu().numpy())
        power = np.concatenate(predicted, axis=0) * float(stats["power_cutoff"])
        power[power < 5.0] = 0.0
        power = np.clip(power, 0.0, float(stats["power_cutoff"]))
        power *= power >= float(stats["status_threshold"])
        return np.concatenate(
            [window[:valid] for window, valid in zip(power, valid_lengths)]
        )[: len(aggregate)].astype(np.float32)

    def predict(self, inference_data, context: InferenceContext) -> PredictionOutput:
        if self.normalization is None:
            raise RuntimeError("BERT4NILM must be fitted or loaded before prediction.")
        self.to(context.device)
        self.eval()
        timestamps, households, predictions = [], [], []
        for series in inference_data.series:
            segments = series.segment_ids
            if segments is None:
                segments = np.zeros(len(series.aggregate), dtype=np.int64)
            output = np.empty(len(series.aggregate), dtype=np.float32)
            start = 0
            while start < len(series.aggregate):
                end = start + 1
                while end < len(series.aggregate) and segments[end] == segments[start]:
                    end += 1
                output[start:end] = self._predict_segment(
                    series.aggregate[start:end], context
                )
                start = end
            timestamps.append(series.timestamps)
            households.append(np.repeat(series.household_id, len(series.timestamps)))
            predictions.append(output[:, None])
        return PredictionOutput(
            np.concatenate(timestamps),
            np.concatenate(predictions),
            tuple(self.normalization["appliances"]),
            np.concatenate(households),
            metadata={"inference_window_stride": self.window_size},
        )
