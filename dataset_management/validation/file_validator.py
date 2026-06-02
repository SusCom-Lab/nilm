from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import pandas as pd

from dataset_management.validation.rules import (
    DEFAULT_VALIDATION_THRESHOLDS,
    FILENAME_PATTERN,
    ISSUE_ERROR,
    ISSUE_INFO,
    ISSUE_WARN,
)


def _load_h5_frame(file_path: str) -> pd.DataFrame:
    return pd.read_hdf(file_path, key="dataset")


def _identify_signal_column(df: pd.DataFrame) -> str:
    excluded = {"time", "is_original", "is_filled_short", "is_valid_point"}
    candidates = [col for col in df.columns if col not in excluded]
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one signal column, found {candidates}.")
    return candidates[0]


class FileValidator:
    def __init__(self, thresholds: Dict[str, float] | None = None):
        self.thresholds = dict(DEFAULT_VALIDATION_THRESHOLDS)
        if thresholds:
            self.thresholds.update(thresholds)

    def validate_file(self, file_path: str) -> Dict[str, object]:
        report: Dict[str, object] = {
            "file": file_path,
            "status": ISSUE_INFO,
            "issues": [],
            "stats": {},
        }

        basename = os.path.basename(file_path)
        filename_match = FILENAME_PATTERN.match(basename)
        if not filename_match:
            report["status"] = ISSUE_ERROR
            report["issues"].append({"code": "INVALID_FILENAME", "severity": ISSUE_ERROR})
            return report

        report["house_id"] = filename_match.group("house")
        report["signal_name"] = filename_match.group("signal").lower()

        try:
            df = _load_h5_frame(file_path)
        except Exception as exc:
            report["status"] = ISSUE_ERROR
            report["issues"].append({"code": "READ_FAILED", "severity": ISSUE_ERROR, "message": str(exc)})
            return report

        if df.empty:
            report["status"] = ISSUE_ERROR
            report["issues"].append({"code": "EMPTY_FILE", "severity": ISSUE_ERROR})
            return report

        if "time" not in df.columns:
            report["status"] = ISSUE_ERROR
            report["issues"].append({"code": "MISSING_TIME_COLUMN", "severity": ISSUE_ERROR})
            return report

        try:
            signal_column = _identify_signal_column(df)
        except ValueError as exc:
            report["status"] = ISSUE_ERROR
            report["issues"].append({"code": "INVALID_SIGNAL_COLUMNS", "severity": ISSUE_ERROR, "message": str(exc)})
            return report

        if signal_column.lower() != str(report["signal_name"]):
            report["issues"].append(
                {
                    "code": "SIGNAL_NAME_MISMATCH",
                    "severity": ISSUE_WARN,
                    "expected": report["signal_name"],
                    "actual": signal_column,
                }
            )

        time_series = pd.to_datetime(df["time"], errors="coerce")
        if time_series.isna().all():
            report["status"] = ISSUE_ERROR
            report["issues"].append({"code": "UNPARSEABLE_TIME", "severity": ISSUE_ERROR})
            return report

        signal_values = pd.to_numeric(df[signal_column], errors="coerce")
        time_diffs = time_series.sort_values().diff().dropna().dt.total_seconds()

        row_count = int(len(df))
        duplicate_count = int(time_series.duplicated().sum())
        nan_ratio = float(signal_values.isna().mean())
        negative_ratio = float((signal_values < 0).fillna(False).mean())
        zero_ratio = float((signal_values == 0).fillna(False).mean())
        max_gap_seconds = float(time_diffs.max()) if not time_diffs.empty else 0.0
        median_gap_seconds = float(time_diffs.median()) if not time_diffs.empty else 0.0
        monotonic = bool(time_series.is_monotonic_increasing)
        max_abs_power = float(np.nanmax(np.abs(signal_values.to_numpy(dtype=np.float64)))) if not signal_values.isna().all() else 0.0

        report["stats"] = {
            "row_count": row_count,
            "duplicate_timestamp_count": duplicate_count,
            "nan_ratio": nan_ratio,
            "negative_ratio": negative_ratio,
            "zero_ratio": zero_ratio,
            "max_gap_seconds": max_gap_seconds,
            "median_gap_seconds": median_gap_seconds,
            "monotonic_increasing": monotonic,
            "max_abs_power": max_abs_power,
            "min_timestamp": None if time_series.isna().all() else str(time_series.min()),
            "max_timestamp": None if time_series.isna().all() else str(time_series.max()),
        }

        if not monotonic:
            report["issues"].append({"code": "NON_MONOTONIC_TIME", "severity": ISSUE_WARN})
        if duplicate_count > 0:
            report["issues"].append({"code": "DUPLICATE_TIMESTAMP", "severity": ISSUE_WARN, "count": duplicate_count})
        if nan_ratio > self.thresholds["max_nan_ratio_warn"]:
            report["issues"].append({"code": "HIGH_NAN_RATIO", "severity": ISSUE_WARN, "ratio": nan_ratio})
        if negative_ratio > self.thresholds["max_negative_ratio_warn"]:
            report["issues"].append({"code": "NEGATIVE_POWER", "severity": ISSUE_WARN, "ratio": negative_ratio})
        if median_gap_seconds > 0 and max_gap_seconds > median_gap_seconds * self.thresholds["max_gap_multiplier_warn"]:
            report["issues"].append({"code": "LARGE_GAP", "severity": ISSUE_WARN, "seconds": max_gap_seconds})
        if max_abs_power > self.thresholds["max_abs_power_warn"]:
            report["issues"].append({"code": "EXTREME_POWER_VALUE", "severity": ISSUE_WARN, "value": max_abs_power})

        severities = [issue["severity"] for issue in report["issues"]]
        if ISSUE_ERROR in severities:
            report["status"] = ISSUE_ERROR
        elif ISSUE_WARN in severities:
            report["status"] = ISSUE_WARN

        return report

    def validate_directory(self, root_dir: str) -> List[Dict[str, object]]:
        reports: List[Dict[str, object]] = []
        for current_root, _, files in os.walk(root_dir):
            for name in sorted(files):
                if name.lower().endswith(".h5"):
                    reports.append(self.validate_file(os.path.join(current_root, name)))
        return reports
