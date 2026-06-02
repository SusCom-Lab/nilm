from __future__ import annotations

import os
from typing import Dict, List

import pandas as pd

from dataset_management.export.segment_builder import SegmentBuilder
from dataset_management.repair.repair_config import DATASET_REPAIR_CONFIG
from dataset_management.repair.repair_utils import identify_signal_column


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
        merged = merged[["time", "aggregate", appliance_name, "segment_id"]]

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
