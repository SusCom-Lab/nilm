from __future__ import annotations

import os
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd


HOUSE_PATTERN = re.compile(r"_H(?P<house>\d+)\.csv$", re.IGNORECASE)


@dataclass(frozen=True)
class ClassicalGroup:
    group_id: str
    time: pd.Series
    aggregate: np.ndarray
    appliances: dict[str, np.ndarray]


@dataclass(frozen=True)
class ClassicalInferenceGroup:
    """Joint-model input that deliberately excludes appliance labels."""

    group_id: str
    time: pd.Series
    aggregate: np.ndarray


def hide_classical_targets(
    grouped_data: dict[str, ClassicalGroup],
) -> dict[str, ClassicalInferenceGroup]:
    """Create target-free views for model-owned joint disaggregation."""

    return {
        group_id: ClassicalInferenceGroup(
            group_id=group.group_id,
            time=group.time,
            aggregate=group.aggregate,
        )
        for group_id, group in grouped_data.items()
    }


def parse_house_id(csv_path: str) -> str:
    match = HOUSE_PATTERN.search(os.path.basename(csv_path))
    if match is None:
        raise ValueError(f"Could not parse house id from filename: {csv_path}")
    return f"H{match.group('house')}"


def _load_single_appliance_csv(csv_path: str) -> tuple[str, pd.DataFrame]:
    df = pd.read_csv(csv_path, low_memory=False)
    if df.shape[1] < 3:
        raise ValueError(f"Expected at least 3 columns in {csv_path}, found {df.shape[1]}.")

    time_col = df.columns[0]
    aggregate_col = df.columns[1]
    appliance_col = df.columns[2]
    appliance_name = str(appliance_col)

    loaded = pd.DataFrame(
        {
            "time": df[time_col],
            f"aggregate__{appliance_name}": pd.to_numeric(df[aggregate_col], errors="coerce"),
            appliance_name: pd.to_numeric(df[appliance_col], errors="coerce"),
        }
    ).dropna()
    if loaded.empty:
        raise ValueError(f"CSV {csv_path} does not contain valid rows after numeric conversion.")
    return appliance_name, loaded


def load_grouped_classical_data(
    csv_paths: list[str],
    *,
    min_appliances: int = 2,
    aggregate_tolerance: float = 1e-3,
) -> dict[str, ClassicalGroup]:
    grouped_paths: dict[str, list[str]] = {}
    for path in csv_paths:
        house_id = parse_house_id(path)
        grouped_paths.setdefault(house_id, []).append(path)

    grouped_data: dict[str, ClassicalGroup] = {}
    for group_id, paths in grouped_paths.items():
        if len(paths) < min_appliances:
            raise ValueError(
                f"Group {group_id} must contain at least {min_appliances} appliance CSVs, found {len(paths)}."
            )

        merged: pd.DataFrame | None = None
        aggregate_columns: list[str] = []
        appliance_names: list[str] = []
        for path in sorted(paths):
            appliance_name, df = _load_single_appliance_csv(path)
            appliance_names.append(appliance_name)
            aggregate_columns.append(f"aggregate__{appliance_name}")
            merged = df if merged is None else merged.merge(df, on="time", how="inner")

        if merged is None:
            raise ValueError(f"Unable to build aligned classical group for {group_id}.")

        if merged.empty:
            raise ValueError(
                "No overlapping timestamps found for "
                f"{group_id}. Refusing benchmark-incompatible positional fallback."
            )

        base_aggregate = merged[aggregate_columns[0]].to_numpy(dtype=np.float32)
        for column in aggregate_columns[1:]:
            current = merged[column].to_numpy(dtype=np.float32)
            if not np.allclose(base_aggregate, current, atol=aggregate_tolerance, rtol=0.0):
                max_delta = float(np.max(np.abs(base_aggregate - current)))
                raise ValueError(
                    f"Aggregate mismatch detected in {group_id}; max absolute difference is {max_delta:.6f}."
                )

        appliances = {
            appliance_name: merged[appliance_name].to_numpy(dtype=np.float32)
            for appliance_name in appliance_names
        }
        grouped_data[group_id] = ClassicalGroup(
            group_id=group_id,
            time=merged["time"].reset_index(drop=True),
            aggregate=base_aggregate,
            appliances=appliances,
        )

    return grouped_data
