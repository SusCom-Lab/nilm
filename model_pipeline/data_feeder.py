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
        split_ratio=None,
        split_mode=None,
        target_mode="point",
        output_size=None,
        output_offset=None,
        normalisation_stats=None,
    ):
        self.window_size = int(window_size)
        self.target_mode = target_mode
        self.output_size = int(output_size or (1 if target_mode == "point" else self.window_size))
        self.center_index = self.window_size // 2
        self.output_offset = self._resolve_output_offset(output_offset)
        self.data = []
        self.normalisation_params = {}
        self.normalisation_stats = None if normalisation_stats is None else dict(normalisation_stats)
        self.window_locations = []

        split_dfs = []
        for file in file_dirs:
            print(f"Loading data: {os.path.basename(file)} ...")
            df = pd.read_csv(file)
            print(f"  Loaded {len(df)} rows.")
            if crop:
                df = df.head(crop)
            split_df = self._select_split(df, split_ratio, split_mode)
            split_dfs.append((file, split_df.reset_index(drop=True)))

        if self.normalisation_stats is None:
            self.normalisation_stats = self._compute_normalisation_stats(
                [df for _, df in split_dfs if not df.empty]
            )

        for file, df in split_dfs:
            print(f"  Normalising {len(df)} rows.")
            normalised_df = df.copy()
            aggregate_column = normalised_df.columns[1]
            appliance_column = normalised_df.columns[2]
            normalised_df[aggregate_column] = pd.to_numeric(normalised_df[aggregate_column], errors="coerce").astype(np.float32)
            normalised_df[appliance_column] = pd.to_numeric(normalised_df[appliance_column], errors="coerce").astype(np.float32)
            stats = dict(self.normalisation_stats)
            normalised_df.iloc[:, 1] = (normalised_df.iloc[:, 1] - stats["aggregate_mean"]) / stats["aggregate_std"]
            normalised_df.iloc[:, 2] = (normalised_df.iloc[:, 2] - stats["appliance_mean"]) / stats["appliance_std"]

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

                inputs = torch.tensor(segment_df.iloc[:, 1].to_numpy(), dtype=torch.float32)
                outputs = torch.tensor(segment_df.iloc[:, 2].to_numpy(), dtype=torch.float32)
                self.data.append((inputs, outputs, segment_indices))
                self.window_locations.extend(
                    int(segment_indices[start_idx])
                    for start_idx in range(self._num_windows(inputs))
                )

    def _select_split(self, df, split_ratio, split_mode):
        if split_ratio is None or split_mode is None:
            return df.copy()

        if "segment_id" in df.columns:
            segment_ids = df["segment_id"].drop_duplicates()
            split_index = int(len(segment_ids) * split_ratio)
            if split_mode == "train":
                selected_segments = segment_ids.iloc[:split_index]
            elif split_mode == "val":
                selected_segments = segment_ids.iloc[split_index:]
            else:
                return df.copy()
            return df[df["segment_id"].isin(selected_segments)].copy()

        total_rows = len(df)
        split_index = int(total_rows * split_ratio)
        if split_mode == "train":
            return df.iloc[:split_index].copy()
        if split_mode == "val":
            return df.iloc[split_index:].copy()
        return df.copy()

    def _safe_std(self, value):
        value = float(value)
        if not np.isfinite(value) or value == 0.0:
            return 1.0
        return value

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

    def get_window_locations(self):
        return list(self.window_locations)

    def __getitem__(self, idx):
        for inputs, outputs, _ in self.data:
            num_windows = self._num_windows(inputs)
            if idx < num_windows:
                start_idx = idx
                end_idx = idx + self.window_size
                return inputs[start_idx:end_idx], self._slice_targets(outputs, start_idx)
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
