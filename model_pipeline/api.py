"""Stable experiment contracts for decoupled NILM model plugins.

This module intentionally contains no Dataset, DataLoader, loss, optimiser,
scheduler, training loop, or inference-window implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

import numpy as np


OFFICIAL_EXPERIMENT_SEEDS = (42, 3407, 2026)


@dataclass(frozen=True)
class HouseholdSplit:
    csv_paths: tuple[str, ...]
    household_ids: tuple[str, ...]
    name: str

    def __post_init__(self) -> None:
        if not self.csv_paths:
            raise ValueError(f"The {self.name} household split cannot be empty.")
        if len(self.csv_paths) != len(self.household_ids):
            raise ValueError("csv_paths and household_ids must have the same length.")


@dataclass(frozen=True)
class SupervisedSeries:
    """One household's canonical raw training series, including labels."""

    timestamps: np.ndarray
    aggregate: np.ndarray
    appliance_power: np.ndarray
    appliances: tuple[str, ...]
    household_id: str
    segment_ids: np.ndarray | None = None
    status: np.ndarray | None = None

    def __post_init__(self) -> None:
        length = len(self.timestamps)
        if len(self.aggregate) != length or len(self.appliance_power) != length:
            raise ValueError("Supervised arrays must have the same time dimension.")
        if self.appliance_power.ndim != 2:
            raise ValueError("appliance_power must have shape [time, appliance].")
        if self.appliance_power.shape[1] != len(self.appliances):
            raise ValueError("appliance_power columns must match appliances.")
        if self.status is not None and self.status.shape != self.appliance_power.shape:
            raise ValueError("status must match appliance_power shape.")


@dataclass(frozen=True)
class SupervisedPartition:
    name: str
    household_ids: tuple[str, ...]
    series: tuple[SupervisedSeries, ...]

    def __post_init__(self) -> None:
        if not self.series:
            raise ValueError(f"The {self.name} partition cannot be empty.")


@dataclass(frozen=True)
class InferenceSeries:
    """Target-free validation or test input visible to a model plugin."""

    timestamps: np.ndarray
    aggregate: np.ndarray
    appliances: tuple[str, ...]
    household_id: str
    segment_ids: np.ndarray | None = None

    def __post_init__(self) -> None:
        if len(self.timestamps) != len(self.aggregate):
            raise ValueError("Inference timestamps and aggregate must have equal length.")
        if self.segment_ids is not None and len(self.segment_ids) != len(self.aggregate):
            raise ValueError("Inference segment_ids must match aggregate length.")


@dataclass(frozen=True)
class InferencePartition:
    name: str
    household_ids: tuple[str, ...]
    series: tuple[InferenceSeries, ...]

    def __post_init__(self) -> None:
        if not self.series:
            raise ValueError(f"The {self.name} inference partition cannot be empty.")


@dataclass(frozen=True)
class EvaluationTarget:
    """Hidden labels consumed only by the public validator or Evaluator."""

    timestamps: np.ndarray
    appliance_power: np.ndarray
    appliances: tuple[str, ...]
    household_ids: np.ndarray
    status: np.ndarray | None = None

    def __post_init__(self) -> None:
        length = len(self.timestamps)
        if self.appliance_power.shape != (length, len(self.appliances)):
            raise ValueError("Evaluation power must have shape [time, appliance].")
        if len(self.household_ids) != length:
            raise ValueError("Evaluation household_ids must match timestamps.")
        if self.status is not None and self.status.shape != self.appliance_power.shape:
            raise ValueError("Evaluation status must match power shape.")


@dataclass(frozen=True)
class PredictionOutput:
    """Complete protocol-timeline prediction in raw watts."""

    timestamps: np.ndarray
    power: np.ndarray
    appliances: tuple[str, ...]
    household_ids: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        length = len(self.timestamps)
        if self.power.shape != (length, len(self.appliances)):
            raise ValueError("Prediction power must have shape [time, appliance].")
        if len(self.household_ids) != length:
            raise ValueError("Prediction household_ids must match timestamps.")
        if not np.isfinite(self.power).all():
            raise ValueError("Prediction power cannot contain NaN or infinity.")


@dataclass(frozen=True)
class ValidationResult:
    epoch: int
    mse: float
    improved: bool
    should_stop: bool
    checkpoint_path: str | None


@dataclass(frozen=True)
class TrainingContext:
    device: str
    seed: int
    num_epochs: int
    validate_candidate: Callable[..., ValidationResult]


@dataclass(frozen=True)
class InferenceContext:
    device: str
    batch_size: int


@dataclass
class FitResult:
    history: list[dict[str, Any]] = field(default_factory=list)
    best_epoch: int | None = None
    best_validation_mse: float | None = None
    checkpoint_path: str | None = None


@runtime_checkable
class NILMModelPlugin(Protocol):
    display_name: str

    def fit(
        self,
        train_data: SupervisedPartition,
        validation_data: InferencePartition,
        context: TrainingContext,
    ) -> FitResult: ...

    def predict(
        self,
        inference_data: InferencePartition,
        context: InferenceContext,
    ) -> PredictionOutput: ...

    def save(self, path: str, *, metadata: dict[str, Any] | None = None) -> None: ...

    def load(self, path: str, device: str) -> None: ...


__all__ = [
    "EvaluationTarget",
    "FitResult",
    "HouseholdSplit",
    "InferenceContext",
    "InferencePartition",
    "InferenceSeries",
    "NILMModelPlugin",
    "OFFICIAL_EXPERIMENT_SEEDS",
    "PredictionOutput",
    "SupervisedPartition",
    "SupervisedSeries",
    "TrainingContext",
    "ValidationResult",
]
