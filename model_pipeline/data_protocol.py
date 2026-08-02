"""Household-level data partitioning for NILM experiments."""

from __future__ import annotations

import os
import re
from collections.abc import Iterable

from model_pipeline.contracts import HouseholdSplit


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


__all__ = [
    "HOUSEHOLD_PATTERN",
    "build_train_validation_splits",
    "household_id_from_path",
    "make_household_split",
    "validate_disjoint_households",
]
