"""Canonical raw-data loading and household partition enforcement."""

from __future__ import annotations

import os
import re
from collections.abc import Iterable

import numpy as np
import pandas as pd

from model_pipeline.api import (
    EvaluationTarget,
    HouseholdSplit,
    InferencePartition,
    InferenceSeries,
    SupervisedPartition,
    SupervisedSeries,
)


HOUSEHOLD_PATTERN = re.compile(r"(?:^|_)H(?P<house>\d+)(?:_|\.|$)", re.IGNORECASE)


def household_id_from_path(csv_path: str) -> str:
    """Return the canonical ``H<number>`` identifier encoded in a CSV name."""

    filename = os.path.basename(os.fspath(csv_path))
    match = HOUSEHOLD_PATTERN.search(filename)
    if match is None:
        raise ValueError(
            f"Cannot determine household from '{filename}'; expected a name containing H<number>."
        )
    return f"H{int(match.group('house'))}"


def make_household_split(name: str, csv_paths: Iterable[str]) -> HouseholdSplit:
    paths = tuple(os.fspath(path) for path in csv_paths)
    household_ids = tuple(household_id_from_path(path) for path in paths)
    return HouseholdSplit(csv_paths=paths, household_ids=household_ids, name=name)


def validate_disjoint_households(*splits: HouseholdSplit) -> None:
    """Reject household leakage across any pair of experiment partitions."""

    for left_index, left in enumerate(splits):
        left_houses = set(left.household_ids)
        for right in splits[left_index + 1 :]:
            overlap = sorted(left_houses.intersection(right.household_ids))
            if overlap:
                raise ValueError(
                    f"Household leakage between {left.name} and {right.name}: "
                    f"{', '.join(overlap)}."
                )


def build_train_validation_splits(
    train_csv_paths: Iterable[str],
    validation_csv_paths: Iterable[str],
) -> tuple[HouseholdSplit, HouseholdSplit]:
    train = make_household_split("training", train_csv_paths)
    validation = make_household_split("validation", validation_csv_paths)
    validate_disjoint_households(train, validation)
    return train, validation


def _load_appliance_csv(csv_path: str, crop: int | None = None) -> dict:
    frame = pd.read_csv(csv_path, low_memory=False)
    if crop is not None:
        frame = frame.iloc[: int(crop)].copy()
    if frame.shape[1] < 3:
        raise ValueError(f"Expected timestamp, aggregate, and appliance columns in {csv_path}.")

    timestamp_column, aggregate_column, appliance_column = frame.columns[:3]
    timestamps = pd.to_datetime(frame[timestamp_column], errors="coerce")
    aggregate = pd.to_numeric(frame[aggregate_column], errors="coerce")
    appliance = pd.to_numeric(frame[appliance_column], errors="coerce")
    valid = timestamps.notna() & aggregate.notna() & appliance.notna()
    if not bool(valid.all()):
        raise ValueError(f"Invalid timestamp or power values found in {csv_path}.")
    if frame.empty:
        raise ValueError(f"CSV {csv_path} is empty.")

    segment_ids = (
        frame["segment_id"].to_numpy()
        if "segment_id" in frame.columns
        else np.zeros(len(frame), dtype=np.int64)
    )
    status = (
        pd.to_numeric(frame["status"], errors="raise").to_numpy(dtype=np.float32)
        if "status" in frame.columns
        else None
    )
    return {
        "timestamps": timestamps.to_numpy(),
        "aggregate": aggregate.to_numpy(dtype=np.float32),
        "appliance": appliance.to_numpy(dtype=np.float32),
        "appliance_name": str(appliance_column),
        "segment_ids": segment_ids,
        "status": status,
        "source": os.fspath(csv_path),
    }


def load_supervised_partition(
    name: str,
    csv_paths: Iterable[str],
    *,
    crop: int | None = None,
) -> SupervisedPartition:
    """Load fixed raw series and align multi-appliance files by household/time."""

    paths = tuple(os.fspath(path) for path in csv_paths)
    split = make_household_split(name, paths)
    grouped: dict[str, list[dict]] = {}
    for path, household_id in zip(paths, split.household_ids):
        grouped.setdefault(household_id, []).append(_load_appliance_csv(path, crop=crop))

    series_items: list[SupervisedSeries] = []
    expected_appliances: tuple[str, ...] | None = None
    for household_id in sorted(grouped, key=lambda value: int(value[1:])):
        sources = sorted(grouped[household_id], key=lambda item: item["appliance_name"])
        base = pd.DataFrame(
            {
                "timestamp": sources[0]["timestamps"],
                "aggregate": sources[0]["aggregate"],
                "segment_id": sources[0]["segment_ids"],
            }
        )
        appliance_names: list[str] = []
        status_columns: list[str] = []
        for index, source in enumerate(sources):
            appliance_name = source["appliance_name"]
            if appliance_name in appliance_names:
                raise ValueError(f"Duplicate appliance '{appliance_name}' for {household_id}.")
            appliance_names.append(appliance_name)
            source_frame = pd.DataFrame(
                {
                    "timestamp": source["timestamps"],
                    f"aggregate_{index}": source["aggregate"],
                    appliance_name: source["appliance"],
                }
            )
            status_name = f"status_{appliance_name}"
            if source["status"] is not None:
                source_frame[status_name] = source["status"]
                status_columns.append(status_name)
            base = base.merge(source_frame, on="timestamp", how="inner")

        if base.empty:
            raise ValueError(f"No aligned timestamps remain for household {household_id}.")
        for index in range(len(sources)):
            current = base[f"aggregate_{index}"].to_numpy(dtype=np.float32)
            if not np.allclose(base["aggregate"], current, atol=1e-3, rtol=0.0):
                raise ValueError(f"Aggregate mismatch across appliance files for {household_id}.")

        appliances = tuple(appliance_names)
        if expected_appliances is None:
            expected_appliances = appliances
        elif appliances != expected_appliances:
            raise ValueError(
                f"All households in {name} must contain the same appliances; "
                f"expected {expected_appliances}, got {appliances} for {household_id}."
            )
        status = None
        if len(status_columns) == len(appliances):
            status = base[status_columns].to_numpy(dtype=np.float32)
        series_items.append(
            SupervisedSeries(
                timestamps=base["timestamp"].to_numpy(),
                aggregate=base["aggregate"].to_numpy(dtype=np.float32),
                appliance_power=base[list(appliances)].to_numpy(dtype=np.float32),
                appliances=appliances,
                household_id=household_id,
                segment_ids=base["segment_id"].to_numpy(),
                status=status,
            )
        )

    return SupervisedPartition(
        name=name,
        household_ids=tuple(item.household_id for item in series_items),
        series=tuple(series_items),
    )


def hide_partition_targets(partition: SupervisedPartition) -> InferencePartition:
    return InferencePartition(
        name=partition.name,
        household_ids=partition.household_ids,
        series=tuple(
            InferenceSeries(
                timestamps=item.timestamps,
                aggregate=item.aggregate,
                appliances=item.appliances,
                household_id=item.household_id,
                segment_ids=item.segment_ids,
            )
            for item in partition.series
        ),
    )


def build_evaluation_target(partition: SupervisedPartition) -> EvaluationTarget:
    appliances = partition.series[0].appliances
    if any(item.appliances != appliances for item in partition.series):
        raise ValueError("Evaluation series must use the same appliance ordering.")
    statuses = [item.status for item in partition.series]
    status = None
    if all(value is not None for value in statuses):
        status = np.concatenate(statuses, axis=0)
    return EvaluationTarget(
        timestamps=np.concatenate([item.timestamps for item in partition.series]),
        appliance_power=np.concatenate(
            [item.appliance_power for item in partition.series], axis=0
        ),
        appliances=appliances,
        household_ids=np.concatenate(
            [np.repeat(item.household_id, len(item.timestamps)) for item in partition.series]
        ),
        status=status,
    )


def validate_experiment_partitions(
    train: SupervisedPartition,
    validation: SupervisedPartition,
    test: SupervisedPartition | None = None,
) -> None:
    splits = [
        HouseholdSplit(tuple("unused" for _ in train.household_ids), train.household_ids, "training"),
        HouseholdSplit(
            tuple("unused" for _ in validation.household_ids),
            validation.household_ids,
            "validation",
        ),
    ]
    if test is not None:
        splits.append(
            HouseholdSplit(tuple("unused" for _ in test.household_ids), test.household_ids, "test")
        )
    validate_disjoint_households(*splits)


__all__ = [
    "HOUSEHOLD_PATTERN",
    "build_train_validation_splits",
    "build_evaluation_target",
    "household_id_from_path",
    "hide_partition_targets",
    "load_supervised_partition",
    "make_household_split",
    "validate_disjoint_households",
    "validate_experiment_partitions",
]
