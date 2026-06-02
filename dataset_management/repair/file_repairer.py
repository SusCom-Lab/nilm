from __future__ import annotations

import os
from typing import Dict, List

import pandas as pd

from dataset_management.repair.repair_config import DATASET_REPAIR_CONFIG
from dataset_management.repair.repair_utils import (
    clip_power_values,
    deduplicate_by_time,
    forward_fill_limited,
    identify_signal_column,
    resample_signal,
)


class FileRepairer:
    def __init__(self, config: Dict[str, Dict[str, object]] | None = None):
        self.config = config or DATASET_REPAIR_CONFIG

    def _get_dataset_config(self, dataset_type: str) -> Dict[str, object]:
        dataset_key = dataset_type.upper()
        if dataset_key not in self.config:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")
        return self.config[dataset_key]

    def repair_file(self, file_path: str, dataset_type: str, save_path: str) -> Dict[str, object]:
        cfg = self._get_dataset_config(dataset_type)
        df = pd.read_hdf(file_path, key="dataset")
        signal_column = identify_signal_column(df)

        df = df[["time", signal_column]].copy()
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.dropna(subset=["time"]).sort_values("time")
        df = deduplicate_by_time(df, signal_column)
        df = resample_signal(df, signal_column, str(cfg["resample_interval"]))

        step = pd.Timedelta(str(cfg["resample_interval"]))
        max_fill = pd.Timedelta(str(cfg["max_fill_duration"]))
        fill_limit = int(max_fill / step)

        original_series = pd.to_numeric(df[signal_column], errors="coerce")
        filled_series = forward_fill_limited(original_series, fill_limit)

        repaired = pd.DataFrame({"time": df["time"]})
        repaired[signal_column] = clip_power_values(
            filled_series,
            float(cfg["min_power_threshold"]),
            float(cfg["max_power_clip"]),
        )
        repaired["is_original"] = original_series.notna()
        repaired["is_filled_short"] = original_series.isna() & filled_series.notna()
        repaired["is_valid_point"] = repaired["is_original"] | repaired["is_filled_short"]

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        repaired.to_hdf(save_path, key="dataset", mode="w", format="table")

        return {
            "input_file": file_path,
            "output_file": save_path,
            "signal_name": signal_column,
            "row_count": int(len(repaired)),
            "valid_points": int(repaired["is_valid_point"].sum()),
            "filled_short_points": int(repaired["is_filled_short"].sum()),
            "resample_interval": str(cfg["resample_interval"]),
            "max_fill_duration": str(cfg["max_fill_duration"]),
        }

    def repair_directory(self, input_dir: str, dataset_type: str, output_dir: str) -> List[Dict[str, object]]:
        summaries: List[Dict[str, object]] = []
        for current_root, _, files in os.walk(input_dir):
            rel_root = os.path.relpath(current_root, input_dir)
            destination_root = output_dir if rel_root == "." else os.path.join(output_dir, rel_root)
            for name in sorted(files):
                if not name.lower().endswith(".h5"):
                    continue
                input_file = os.path.join(current_root, name)
                output_file = os.path.join(destination_root, name)
                summaries.append(self.repair_file(input_file, dataset_type, output_file))
        return summaries
