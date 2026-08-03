"""Official point-output RNN NILM plugin.

Official repository: https://github.com/nilmtk/nilmtk-contrib
Official revision: 14efd545e09d836e02159b23444350af92c6ee70
Official file: nilmtk_contrib/torch/rnn.py
Local adaptations: fixed raw partitions, public validation MSE, complete
protocol-timeline prediction, and repository checkpoint metadata.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from model_pipeline.api import (
    FitResult,
    InferenceContext,
    InferencePartition,
    PredictionOutput,
    TrainingContext,
)
from model_pipeline.model_registry import register_model


class RNNDataset(Dataset):
    def __init__(self, partition, window_size, stats):
        if len(partition.series[0].appliances) != 1:
            raise ValueError("RNN is a single-appliance model.")
        self.inputs = []
        self.targets = []
        half = window_size // 2
        for series in partition.series:
            segments = series.segment_ids
            if segments is None:
                segments = np.zeros(len(series.aggregate), dtype=np.int64)
            start = 0
            while start < len(series.aggregate):
                end = start + 1
                while end < len(series.aggregate) and segments[end] == segments[start]:
                    end += 1
                padded = np.pad(series.aggregate[start:end], (half, half))
                for index in range(end - start):
                    self.inputs.append(
                        torch.as_tensor(
                            (padded[index : index + window_size] - stats["mains_mean"])
                            / stats["mains_std"],
                            dtype=torch.float32,
                        )
                    )
                    self.targets.append(
                        torch.tensor(
                            (series.appliance_power[start + index, 0] - stats["appliance_mean"])
                            / stats["appliance_std"],
                            dtype=torch.float32,
                        )
                    )
                start = end

    def __len__(self): return len(self.inputs)
    def __getitem__(self, index): return self.inputs[index], self.targets[index]


@register_model("rnn", aliases=("RNN",), display_name="RNN")
class RNNBaseline(nn.Module):
    display_name = "RNN"
    model_family = "rnn"
    target_type = "point"
    supports_gradient = True
    is_joint_classical = False
    default_window_size = 19
    default_num_epochs = 10
    official_batch_size = 512
    official_optimizer_epsilon = 1e-7
    official_gradient_clip_norm = None

    def __init__(self, *, window_size: int = 19):
        super().__init__()
        if window_size < 2:
            raise ValueError("RNN window_size must be at least 2.")
        self.window_size = window_size
        self.output_size = 1
        self.output_offset = window_size // 2
        self._config = {"window_size": window_size}
        self.normalization: dict[str, Any] | None = None
        self.conv1d = nn.Conv1d(1, 16, kernel_size=4, stride=1, padding=2)
        self.lstm1 = nn.LSTM(16, 128, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(256, 256, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 1)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, parameter in module.named_parameters():
                    if "weight" in name:
                        nn.init.xavier_uniform_(parameter)
                    elif "bias" in name:
                        nn.init.zeros_(parameter)

    def forward(self, inputs):
        inputs = inputs.unsqueeze(1)
        inputs = self.conv1d(inputs).permute(0, 2, 1)
        inputs, _ = self.lstm1(inputs)
        inputs, _ = self.lstm2(inputs)
        inputs = torch.tanh(self.fc1(inputs[:, -1, :]))
        return self.fc2(inputs).reshape(-1)

    def _set_normalization(self, train_data):
        targets = np.concatenate([item.appliance_power[:, 0] for item in train_data.series])
        std = float(np.std(targets))
        if not np.isfinite(std) or std < 1.0:
            std = 100.0
        self.normalization = {
            "mains_mean": 1800.0,
            "mains_std": 600.0,
            "appliance_mean": float(np.mean(targets)),
            "appliance_std": std,
            "appliances": train_data.series[0].appliances,
        }

    def fit(self, train_data, validation_data, context: TrainingContext):
        del validation_data
        self._set_normalization(train_data)
        loader = DataLoader(
            RNNDataset(train_data, self.window_size, self.normalization),
            batch_size=self.official_batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(context.seed),
        )
        self.to(context.device)
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=0.001,
            betas=(0.9, 0.999),
            eps=float(self.official_optimizer_epsilon),
        )
        criterion = nn.MSELoss()
        history = []
        best_epoch = None
        best_mse = None
        checkpoint_path = None
        for epoch in range(1, context.num_epochs + 1):
            self.train()
            total = 0.0
            for inputs, targets in loader:
                optimizer.zero_grad()
                loss = criterion(self(inputs.to(context.device)), targets.to(context.device))
                loss.backward()
                if self.official_gradient_clip_norm is not None:
                    nn.utils.clip_grad_norm_(
                        self.parameters(), float(self.official_gradient_clip_norm)
                    )
                optimizer.step()
                total += float(loss.item())
            validation = context.validate_candidate(epoch=epoch, model=self)
            history.append({
                "epoch": epoch,
                "training_loss": total / max(1, len(loader)),
                "validation_mse": validation.mse,
            })
            if validation.improved:
                best_epoch = epoch
                best_mse = validation.mse
                checkpoint_path = validation.checkpoint_path
            if validation.should_stop:
                break
        return FitResult(history, best_epoch, best_mse, checkpoint_path)

    def predict(self, inference_data: InferencePartition, context: InferenceContext):
        if self.normalization is None:
            raise RuntimeError("RNN must be fitted or loaded before prediction.")
        self.to(context.device)
        self.eval()
        stats = self.normalization
        half = self.window_size // 2
        timestamps = []
        households = []
        predictions = []
        with torch.inference_mode():
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
                    padded = np.pad(series.aggregate[start:end], (half, half))
                    windows = np.stack([
                        padded[index : index + self.window_size] for index in range(end - start)
                    ])
                    windows = (windows - stats["mains_mean"]) / stats["mains_std"]
                    batches = []
                    for offset in range(0, len(windows), context.batch_size):
                        inputs = torch.as_tensor(
                            windows[offset : offset + context.batch_size],
                            dtype=torch.float32,
                            device=context.device,
                        )
                        batches.append(self(inputs).cpu().numpy())
                    normalized = np.concatenate(batches)
                    output[start:end] = normalized * stats["appliance_std"] + stats["appliance_mean"]
                    start = end
                timestamps.append(series.timestamps)
                households.append(np.repeat(series.household_id, len(series.timestamps)))
                predictions.append(output[:, None])
        return PredictionOutput(
            np.concatenate(timestamps),
            np.concatenate(predictions),
            tuple(stats["appliances"]),
            np.concatenate(households),
        )

    def save(self, path, *, metadata=None):
        torch.save({
            "model_key": self._registry_key,
            "model_name": self.display_name,
            "init_kwargs": dict(self._config),
            "model_state": self.state_dict(),
            "normalization": self.normalization,
            "metadata": dict(metadata or {}),
        }, path)

    def load(self, path, device):
        try:
            checkpoint = torch.load(path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location=device)
        self.load_state_dict(checkpoint["model_state"])
        self.normalization = checkpoint["normalization"]
        self.to(device)

    def get_window_size(self): return self.window_size
    def getWindowSize(self): return self.window_size
    def get_output_size(self): return self.output_size
    def get_output_offset(self): return self.output_offset
    def get_target_type(self): return self.target_type
    def get_init_kwargs(self): return dict(self._config)

    def freeze_for_finetuning(self):
        for parameter in self.parameters(): parameter.requires_grad = False
        for module in self.modules():
            if isinstance(module, nn.Linear):
                for parameter in module.parameters(): parameter.requires_grad = True
