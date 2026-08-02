"""Stable data contracts shared by the experiment runner and model plugins.

The contracts deliberately describe experiment inputs and outputs only.  They do
not prescribe a Dataset implementation, optimiser, loss, scheduler, or training
loop; those remain part of each model plugin.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class HouseholdSplit:
    """Files assigned to one supervised experiment partition."""

    csv_paths: tuple[str, ...]
    household_ids: tuple[str, ...]
    name: str

    def __post_init__(self) -> None:
        if len(self.csv_paths) == 0:
            raise ValueError(f"The {self.name} household split cannot be empty.")
        if len(self.csv_paths) != len(self.household_ids):
            raise ValueError("csv_paths and household_ids must have the same length.")


@dataclass(frozen=True)
class SupervisedData:
    """A labelled partition plus an optional compatibility DataLoader.

    Source-faithful plugins may build their official Dataset directly from
    ``household_split.csv_paths``.  Existing simple baselines can consume the
    prepared ``loader`` while they migrate to private training recipes.
    """

    household_split: HouseholdSplit | None
    loader: Any | None = None
    normalisation_stats: dict[str, float] | None = None


@dataclass(frozen=True)
class InferenceData:
    """Information available to a model at test time.

    Appliance power and status are intentionally absent.  The evaluator keeps
    those values in a separate :class:`EvaluationTarget`.
    """

    timestamps: np.ndarray
    aggregate: np.ndarray
    household_id: str
    segment_ids: np.ndarray | None = None
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationTarget:
    """Hidden test labels consumed by the evaluator, never by ``predict``."""

    timestamps: np.ndarray
    appliance_power: np.ndarray
    status: np.ndarray | None = None


@dataclass(frozen=True)
class PredictionOutput:
    """A model prediction aligned to explicit timestamps in raw power units."""

    timestamps: np.ndarray
    power: np.ndarray
    valid_mask: np.ndarray | None = None
    status: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.timestamps) != len(self.power):
            raise ValueError("Prediction timestamps and power must have the same length.")
        if self.valid_mask is not None and len(self.valid_mask) != len(self.power):
            raise ValueError("Prediction valid_mask must match the power length.")
        if self.status is not None and len(self.status) != len(self.power):
            raise ValueError("Prediction status must match the power length.")


@dataclass
class FitResult:
    """Training summary returned by every model plugin."""

    best_epoch: int | None = None
    best_score: float | None = None
    history: list[dict[str, Any]] = field(default_factory=list)
    checkpoint_path: str | None = None


@dataclass(frozen=True)
class TrainingContext:
    """Experiment-owned services exposed to a model's private training loop."""

    device: str
    seed: int
    num_epochs: int
    checkpoint_path: Path
    checkpoint_callback: Callable[..., str | None]
    config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InferenceContext:
    device: str
    batch_size: int
    normalisation_stats: dict[str, float] | None = None
    config: dict[str, Any] = field(default_factory=dict)


__all__ = [
    "EvaluationTarget",
    "FitResult",
    "HouseholdSplit",
    "InferenceContext",
    "InferenceData",
    "PredictionOutput",
    "SupervisedData",
    "TrainingContext",
]
