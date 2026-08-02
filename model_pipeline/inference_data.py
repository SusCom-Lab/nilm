"""Target-free window construction used by default model plugins."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class InferenceWindowDataset(Dataset):
    """Build aggregate-only windows without exposing appliance labels."""

    def __init__(
        self,
        *,
        timestamps,
        aggregate,
        segment_ids=None,
        window_size,
        output_size,
        output_offset,
        normalisation_stats,
        include_temporal_features=False,
    ):
        self.timestamps = np.asarray(timestamps)
        aggregate = np.asarray(aggregate, dtype=np.float32)
        if len(self.timestamps) != len(aggregate):
            raise ValueError("Inference timestamps and aggregate must have the same length.")

        stats = dict(normalisation_stats or {})
        mean = float(stats["aggregate_mean"])
        std = float(stats["aggregate_std"])
        if not np.isfinite(std) or std <= 0:
            raise ValueError("Inference requires a finite positive aggregate_std.")
        aggregate = (aggregate - mean) / std

        self.window_size = int(window_size)
        self.output_size = int(output_size)
        self.output_offset = int(output_offset)
        self.include_temporal_features = bool(include_temporal_features)
        self.window_start_indices = []
        self.windows = []

        if segment_ids is None:
            segment_ids = np.zeros(len(aggregate), dtype=np.int64)
        segment_ids = np.asarray(segment_ids)
        if len(segment_ids) != len(aggregate):
            raise ValueError("Inference segment_ids must match the aggregate length.")

        start = 0
        while start < len(aggregate):
            end = start + 1
            while end < len(aggregate) and segment_ids[end] == segment_ids[start]:
                end += 1
            self._append_segment(start, end, aggregate)
            start = end

    def _append_segment(self, start, end, aggregate):
        required = max(self.window_size, self.output_offset + self.output_size)
        count = max(0, (end - start) - required + 1)
        if count == 0:
            return

        temporal = None
        if self.include_temporal_features:
            temporal = self._temporal_features(self.timestamps[start:end])

        for local_start in range(count):
            local_end = local_start + self.window_size
            power_window = torch.from_numpy(
                aggregate[start + local_start : start + local_end].copy()
            )
            if temporal is not None:
                power_window = torch.cat(
                    (power_window.unsqueeze(0), temporal[:, local_start:local_end]),
                    dim=0,
                )
            self.windows.append(power_window)
            self.window_start_indices.append(start + local_start)

    @staticmethod
    def _temporal_features(timestamps):
        values = pd.to_datetime(pd.Series(timestamps), errors="coerce")
        if values.isna().any():
            raise ValueError("Temporal models require valid inference timestamps.")
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
        return torch.from_numpy(np.stack(channels).astype(np.float32, copy=False))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        return self.windows[index]


__all__ = ["InferenceWindowDataset"]
