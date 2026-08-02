from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SlidingWindowDataset(Dataset):
    def __init__(
        self,
        file_dirs,
        window_size,
        crop=None,
        target_mode="point",
        output_size=None,
        output_offset=None,
        normalisation_stats=None,
        appliance_on_threshold=None,
        min_on_points=1,
        min_on_rate=0.0,
        status_column=None,
        include_temporal_features=False,
    ):
        self.window_size = int(window_size)
        self.target_mode = target_mode
        self.output_size = int(output_size or (1 if target_mode == "point" else self.window_size))
        self.center_index = self.window_size // 2
        self.output_offset = self._resolve_output_offset(output_offset)
        self.appliance_on_threshold = appliance_on_threshold
        self.min_on_points = int(min_on_points)
        self.min_on_rate = float(min_on_rate)
        self.status_column = status_column
        self.include_temporal_features = bool(include_temporal_features)
        self.data = []
        self.normalisation_params = {}
        self.normalisation_stats = None if normalisation_stats is None else dict(normalisation_stats)
        self.window_locations = []

        source_dfs = []
        for file in file_dirs:
            print(f"Loading data: {os.path.basename(file)} ...")
            df = pd.read_csv(file)
            print(f"  Loaded {len(df)} rows.")
            if crop:
                df = df.head(crop)
            source_dfs.append((file, df.reset_index(drop=True)))

        if self.normalisation_stats is None:
            self.normalisation_stats = self._compute_normalisation_stats(
                [df for _, df in source_dfs if not df.empty]
            )

        for file, df in source_dfs:
            print(f"  Normalising {len(df)} rows.")
            normalised_df = df.copy()
            aggregate_column = normalised_df.columns[1]
            appliance_column = normalised_df.columns[2]
            if self.status_column is not None and self.status_column not in normalised_df.columns:
                raise ValueError(f"CSV is missing required status column: {self.status_column}")
            status_column = self.status_column
            normalised_df[aggregate_column] = pd.to_numeric(normalised_df[aggregate_column], errors="coerce").astype(np.float32)
            normalised_df[appliance_column] = pd.to_numeric(normalised_df[appliance_column], errors="coerce").astype(np.float32)
            if status_column is not None:
                normalised_df[status_column] = pd.to_numeric(normalised_df[status_column], errors="coerce").fillna(0).astype(np.float32)
            raw_appliance_values = normalised_df[appliance_column].copy()
            stats = dict(self.normalisation_stats)
            normalised_df.iloc[:, 1] = (normalised_df.iloc[:, 1] - stats["aggregate_mean"]) / stats["aggregate_std"]
            normalised_df.iloc[:, 2] = (normalised_df.iloc[:, 2] - stats["appliance_mean"]) / stats["appliance_std"]
            normalised_df["_raw_appliance_for_window_filter"] = raw_appliance_values

            self.normalisation_params[file] = stats
            if "segment_id" in normalised_df.columns:
                grouped_segments = normalised_df.groupby("segment_id", sort=False)
            else:
                grouped_segments = [(0, normalised_df)]

            for _, segment_df in grouped_segments:
                segment_indices = segment_df.index.to_numpy()
                segment_df = segment_df.reset_index(drop=True)
                if len(segment_df) < max(self.window_size, self.output_offset + self.output_size):
                    continue

                raw_appliance_values = segment_df["_raw_appliance_for_window_filter"].to_numpy(dtype=np.float32)
                if not self._segment_passes_on_filter(raw_appliance_values):
                    continue

                inputs = torch.tensor(segment_df.iloc[:, 1].to_numpy(), dtype=torch.float32)
                outputs = torch.tensor(segment_df.iloc[:, 2].to_numpy(), dtype=torch.float32)
                temporal_features = None
                if self.include_temporal_features:
                    temporal_features = self._build_temporal_features(
                        segment_df.iloc[:, 0],
                        source=file,
                    )
                status_outputs = None
                if status_column is not None:
                    status_outputs = torch.tensor(segment_df[status_column].to_numpy(), dtype=torch.float32)
                window_starts = list(range(self._num_windows(inputs)))
                if len(window_starts) == 0:
                    continue

                self.data.append(
                    (
                        inputs,
                        outputs,
                        status_outputs,
                        segment_indices,
                        window_starts,
                        temporal_features,
                    )
                )
                self.window_locations.extend(int(segment_indices[start_idx]) for start_idx in window_starts)

    def _safe_std(self, value):
        value = float(value)
        if not np.isfinite(value) or value == 0.0:
            return 1.0
        return value

    @staticmethod
    def _build_temporal_features(timestamp_values, *, source):
        """Encode real timestamps in the official minute/hour/day/month order."""
        timestamps = pd.to_datetime(timestamp_values, errors="coerce")
        if timestamps.isna().any():
            invalid_count = int(timestamps.isna().sum())
            raise ValueError(
                f"NILMFormer requires valid timestamps in the first CSV column; "
                f"found {invalid_count} invalid value(s) in {source}."
            )

        periodic_values = (
            (timestamps.dt.minute.to_numpy(dtype=np.float32), 60.0),
            (timestamps.dt.hour.to_numpy(dtype=np.float32), 24.0),
            (timestamps.dt.dayofweek.to_numpy(dtype=np.float32), 7.0),
            (timestamps.dt.month.to_numpy(dtype=np.float32), 12.0),
        )
        channels = []
        for values, period in periodic_values:
            phase = 2.0 * np.pi * values / period
            channels.extend((np.sin(phase), np.cos(phase)))
        return torch.from_numpy(np.stack(channels).astype(np.float32, copy=False))

    def _compute_normalisation_stats(self, dfs):
        if not dfs:
            raise ValueError("Cannot compute normalisation statistics from empty data.")

        aggregate_values = pd.concat([df.iloc[:, 1] for df in dfs], ignore_index=True)
        appliance_values = pd.concat([df.iloc[:, 2] for df in dfs], ignore_index=True)
        return {
            "aggregate_mean": float(aggregate_values.mean()),
            "aggregate_std": self._safe_std(aggregate_values.std()),
            "appliance_mean": float(appliance_values.mean()),
            "appliance_std": self._safe_std(appliance_values.std()),
        }

    def _resolve_output_offset(self, output_offset):
        if output_offset is not None:
            return int(output_offset)
        if self.target_mode == "point":
            return self.center_index
        return 0

    def __len__(self):
        return len(self.window_locations)

    def getNormalisationParams(self, file_dir):
        return self.normalisation_params.get(file_dir, dict(self.normalisation_stats))

    def get_normalisation_stats(self):
        return dict(self.normalisation_stats)

    def _slice_targets(self, outputs, start_idx):
        if self.target_mode == "point":
            return outputs[start_idx + self.output_offset]

        target_start = start_idx + self.output_offset
        target_end = target_start + self.output_size
        return outputs[target_start:target_end]

    def _num_windows(self, inputs):
        inputs_length = len(inputs)
        last_required_index = max(self.window_size, self.output_offset + self.output_size)
        return max(0, inputs_length - last_required_index + 1)

    def _segment_passes_on_filter(self, raw_appliance_values):
        if self.appliance_on_threshold is None:
            return True

        threshold = float(self.appliance_on_threshold)
        on_points = int((raw_appliance_values > threshold).sum())
        if on_points < max(1, self.min_on_points):
            return False
        if self.min_on_rate > 0 and on_points / len(raw_appliance_values) < self.min_on_rate:
            return False
        return True

    def get_window_locations(self):
        return list(self.window_locations)

    def __getitem__(self, idx):
        for inputs, outputs, status_outputs, _, window_starts, temporal_features in self.data:
            num_windows = len(window_starts)
            if idx < num_windows:
                start_idx = window_starts[idx]
                end_idx = start_idx + self.window_size
                target = self._slice_targets(outputs, start_idx)
                if status_outputs is not None:
                    status_target = self._slice_targets(status_outputs, start_idx)
                    if self.target_mode == "point":
                        target = torch.stack((target.reshape(()), status_target.reshape(())))
                    else:
                        target = torch.stack((target, status_target), dim=-1)
                input_window = inputs[start_idx:end_idx]
                if temporal_features is not None:
                    input_window = torch.cat(
                        (
                            input_window.unsqueeze(0),
                            temporal_features[:, start_idx:end_idx],
                        ),
                        dim=0,
                    )
                return input_window, target
            idx -= num_windows

        raise IndexError("Index out of range")


def reconstruct_series_from_windows(
    window_predictions: np.ndarray,
    total_length: int,
    *,
    target_mode: str,
    output_offset: int,
    output_size: int,
    window_start_indices: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    predictions = np.asarray(window_predictions, dtype=np.float32)
    if predictions.ndim == 1:
        predictions = predictions[:, None]

    accumulator = np.zeros(total_length, dtype=np.float32)
    counts = np.zeros(total_length, dtype=np.float32)

    if window_start_indices is None:
        window_start_indices = list(range(len(predictions)))

    for start_idx, predicted_window in zip(window_start_indices, predictions):
        if target_mode == "point":
            target_idx = start_idx + output_offset
            if 0 <= target_idx < total_length:
                accumulator[target_idx] += float(predicted_window.reshape(-1)[0])
                counts[target_idx] += 1.0
            continue

        target_start = start_idx + output_offset
        target_end = min(total_length, target_start + output_size)
        valid_length = max(0, target_end - target_start)
        if valid_length == 0:
            continue
        accumulator[target_start:target_end] += predicted_window[:valid_length]
        counts[target_start:target_end] += 1.0

    reconstructed = np.zeros(total_length, dtype=np.float32)
    valid_mask = counts > 0
    reconstructed[valid_mask] = accumulator[valid_mask] / counts[valid_mask]
    return reconstructed, counts
