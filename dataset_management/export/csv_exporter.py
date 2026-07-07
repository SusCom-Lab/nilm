from __future__ import annotations

import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd

from dataset_management.export.segment_builder import SegmentBuilder
from dataset_management.repair.repair_config import DATASET_REPAIR_CONFIG
from dataset_management.repair.repair_utils import identify_signal_column


STATUS_RULES_FILE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "model_pipeline", "appliance_status_rules.json")
)


class CSVExporter:
    def __init__(self, config: Dict[str, Dict[str, object]] | None = None):
        self.config = config or DATASET_REPAIR_CONFIG
        self.segment_builder = SegmentBuilder()

    def _get_dataset_config(self, dataset_type: str) -> Dict[str, object]:
        dataset_key = dataset_type.upper()
        if dataset_key not in self.config:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")
        return self.config[dataset_key]

    def _load_signal_frame(self, file_path: str, value_alias: str, valid_alias: str) -> pd.DataFrame:
        df = pd.read_hdf(file_path, key="dataset")
        signal_column = identify_signal_column(df)
        if "is_valid_point" not in df.columns:
            df["is_valid_point"] = pd.to_numeric(df[signal_column], errors="coerce").notna()
        return pd.DataFrame(
            {
                "time": pd.to_datetime(df["time"], errors="coerce"),
                value_alias: pd.to_numeric(df[signal_column], errors="coerce"),
                valid_alias: df["is_valid_point"].astype(bool),
            }
        )

    def _load_status_rule(self, appliance_name: str) -> Dict[str, object] | None:
        try:
            with open(STATUS_RULES_FILE, "r", encoding="utf-8") as handle:
                rules = json.load(handle)
        except FileNotFoundError:
            return None
        return rules.get(appliance_name.replace(" ", "_").lower())

    def _duration_to_samples(self, duration_seconds: float, sample_period_seconds: float) -> int:
        if duration_seconds <= 0:
            return 0
        if not np.isfinite(sample_period_seconds) or sample_period_seconds <= 0:
            sample_period_seconds = 1.0
        return max(1, int(np.ceil(float(duration_seconds) / float(sample_period_seconds))))

    def _apply_duration_rules(
        self,
        status: np.ndarray,
        rule: Dict[str, object],
        sample_period_seconds: float,
    ) -> np.ndarray:
        filtered = np.asarray(status, dtype=np.int8).copy()
        min_off_samples = self._duration_to_samples(float(rule.get("min_off_duration", 0)), sample_period_seconds)
        min_on_samples = self._duration_to_samples(float(rule.get("min_on_duration", 0)), sample_period_seconds)

        def iter_runs(values: np.ndarray):
            if len(values) == 0:
                return
            start = 0
            current = int(values[0])
            for idx in range(1, len(values)):
                value = int(values[idx])
                if value != current:
                    yield start, idx, current
                    start = idx
                    current = value
            yield start, len(values), current

        if min_off_samples > 0:
            for start, end, value in list(iter_runs(filtered)):
                if value == 0 and start > 0 and end < len(filtered) and (end - start) < min_off_samples:
                    filtered[start:end] = 1

        if min_on_samples > 0:
            for start, end, value in list(iter_runs(filtered)):
                if value == 1 and (end - start) < min_on_samples:
                    filtered[start:end] = 0
        return filtered

    def _build_status_labels(self, frame: pd.DataFrame, appliance_name: str) -> pd.Series:
        rule = self._load_status_rule(appliance_name)
        if rule is None:
            return pd.Series(np.nan, index=frame.index, dtype="float32")

        values = pd.to_numeric(frame[appliance_name], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        min_threshold = float(rule["min_threshold"])
        max_threshold = float(rule.get("max_threshold", np.inf))
        status = ((values >= min_threshold) & (values <= max_threshold)).astype(np.int8)

        timestamps = pd.to_datetime(frame["time"], errors="coerce")
        diffs = timestamps.sort_values().diff().dropna().dt.total_seconds()
        diffs = diffs[diffs > 0]
        sample_period = float(diffs.median()) if not diffs.empty else 1.0
        status = self._apply_duration_rules(status, rule, sample_period)
        return pd.Series(status, index=frame.index, dtype="int8")

    def export_house_appliance(
        self,
        aggregate_file: str,
        appliance_file: str,
        dataset_type: str,
        output_csv: str,
    ) -> Dict[str, object]:
        cfg = self._get_dataset_config(dataset_type)
        aggregate_df = self._load_signal_frame(aggregate_file, "aggregate", "aggregate_is_valid")
        appliance_name = os.path.basename(appliance_file).split("_H", 1)[0]
        appliance_df = self._load_signal_frame(appliance_file, appliance_name, "appliance_is_valid")

        merged = aggregate_df.merge(appliance_df, on="time", how="outer")
        merged = self.segment_builder.build_segments(merged, str(cfg["resample_interval"]))
        merged["status"] = self._build_status_labels(merged, appliance_name)
        merged = merged[["time", "aggregate", appliance_name, "status", "segment_id"]]

        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        merged.to_csv(output_csv, index=False)
        return {
            "output_csv": output_csv,
            "appliance": appliance_name,
            "rows": int(len(merged)),
            "segments": int(merged["segment_id"].nunique()) if not merged.empty else 0,
        }

    def export_house_directory(self, house_dir: str, dataset_type: str, output_dir: str) -> List[Dict[str, object]]:
        summaries: List[Dict[str, object]] = []
        house_name = os.path.basename(house_dir)
        house_number = house_name.split("_", 1)[1]
        aggregate_file = os.path.join(house_dir, f"aggregate_H{house_number}.h5")
        if not os.path.exists(aggregate_file):
            return summaries

        for name in sorted(os.listdir(house_dir)):
            if not name.lower().endswith(".h5") or name.lower().startswith("aggregate_"):
                continue
            appliance_file = os.path.join(house_dir, name)
            appliance_name = name.split("_H", 1)[0]
            output_csv = os.path.join(output_dir, f"{appliance_name}_H{house_number}.csv")
            summaries.append(self.export_house_appliance(aggregate_file, appliance_file, dataset_type, output_csv))
        return summaries
