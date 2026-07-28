"""Leakage-audited data utilities for the REFIT Seq2Point Context-FiLM study."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler


@dataclass(frozen=True)
class BlockSpec:
    house: str
    block: str
    start: str
    end: str
    csv_path: str


@dataclass(frozen=True)
class AccessRecord:
    house: str
    block: str
    purpose: str
    columns: tuple[str, ...]


@dataclass(frozen=True)
class NormalisationStats:
    aggregate_mean: float
    aggregate_std: float
    appliance_mean: float
    appliance_std: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _safe_std(values: np.ndarray) -> float:
    value = float(np.std(values, ddof=1))
    return value if np.isfinite(value) and value > 0 else 1.0


class LeakageGuard:
    """Enforce and record the experiment's label-access policy."""

    TARGET_COLUMNS = ("washing_machine", "status")

    def __init__(self, test_house: str = "H6"):
        self.test_house = test_house
        self.predictions_complete = False
        self.records: list[AccessRecord] = []

    def mark_predictions_complete(self) -> None:
        self.predictions_complete = True

    def check(self, spec: BlockSpec, purpose: str, columns: Sequence[str]) -> None:
        columns_tuple = tuple(columns)
        target_read = any(column in self.TARGET_COLUMNS for column in columns_tuple)
        if purpose in {"context", "normalization_aggregate"} and target_read:
            raise PermissionError(f"{purpose} is aggregate-only, requested {columns_tuple}.")
        if spec.house == self.test_house and purpose in {
            "normalization_aggregate",
            "normalization_appliance",
            "checkpoint_selection",
            "threshold_selection",
        }:
            raise PermissionError(f"{self.test_house} cannot be used for {purpose}.")
        if spec.house == self.test_house and target_read:
            if purpose != "final_metrics_labels" or not self.predictions_complete:
                raise PermissionError(
                    f"{self.test_house} labels are sealed until every prediction is complete."
                )
        self.records.append(AccessRecord(spec.house, spec.block, purpose, columns_tuple))

    def audit_dicts(self) -> list[dict[str, object]]:
        return [asdict(record) for record in self.records]


def read_refit_block(
    spec: BlockSpec,
    *,
    columns: Sequence[str],
    purpose: str,
    guard: LeakageGuard,
    chunksize: int = 250_000,
) -> pd.DataFrame:
    """Read a fixed half-open time block from a sorted REFIT export."""
    required = ["time", *[column for column in columns if column != "time"]]
    if "segment_id" not in required:
        required.append("segment_id")
    required = list(dict.fromkeys(required))
    guard.check(spec, purpose, required)
    start = pd.Timestamp(spec.start)
    end = pd.Timestamp(spec.end)
    parts: list[pd.DataFrame] = []
    found_range = False
    for chunk in pd.read_csv(spec.csv_path, usecols=required, chunksize=chunksize):
        times = pd.to_datetime(chunk["time"], errors="raise")
        if times.iloc[-1] < start:
            continue
        mask = (times >= start) & (times < end)
        if bool(mask.any()):
            selected = chunk.loc[mask].copy()
            selected["time"] = times.loc[mask].to_numpy()
            parts.append(selected)
            found_range = True
        if times.iloc[0] >= end or (found_range and times.iloc[-1] >= end):
            break
    if not parts:
        raise ValueError(f"No data in fixed block {spec.house}-{spec.block}: [{start}, {end}).")
    frame = pd.concat(parts, ignore_index=True)
    frame = frame.sort_values("time", kind="stable").reset_index(drop=True)
    return frame


def compute_normalisation_stats(
    aggregate_blocks: Sequence[pd.DataFrame],
    appliance_b_blocks: Sequence[pd.DataFrame],
    appliance: str = "washing_machine",
) -> NormalisationStats:
    """Aggregate: training A+B. Appliance: training B only."""
    if not aggregate_blocks or not appliance_b_blocks:
        raise ValueError("Normalisation requires non-empty training aggregate and B-label blocks.")
    aggregate = np.concatenate(
        [block["aggregate"].to_numpy(dtype=np.float64) for block in aggregate_blocks]
    )
    target = np.concatenate(
        [block[appliance].to_numpy(dtype=np.float64) for block in appliance_b_blocks]
    )
    return NormalisationStats(
        aggregate_mean=float(np.mean(aggregate)),
        aggregate_std=_safe_std(aggregate),
        appliance_mean=float(np.mean(target)),
        appliance_std=_safe_std(target),
    )


def _valid_window_starts(frame: pd.DataFrame, window_size: int) -> np.ndarray:
    starts: list[np.ndarray] = []
    for _, indices in frame.groupby("segment_id", sort=False).indices.items():
        ordered = np.sort(np.asarray(indices, dtype=np.int64))
        if len(ordered) < window_size:
            continue
        local = np.arange(len(ordered) - window_size + 1, dtype=np.int64)
        starts.append(ordered[local])
    if not starts:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(starts)


def select_uniform_context_windows(
    aggregate_block: pd.DataFrame,
    *,
    stats: NormalisationStats,
    window_size: int,
    context_k: int,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Select aggregate windows uniformly over valid chronological starts."""
    forbidden = {"washing_machine", "status"}.intersection(aggregate_block.columns)
    if forbidden:
        raise ValueError(f"Uniform context selector received forbidden columns: {sorted(forbidden)}")
    starts = _valid_window_starts(aggregate_block, window_size)
    if len(starts) == 0:
        raise ValueError("Context block contains no complete window within one segment.")
    count = min(int(context_k), len(starts))
    positions = np.linspace(0, len(starts) - 1, num=count, dtype=np.int64)
    chosen = starts[positions]
    aggregate = aggregate_block["aggregate"].to_numpy(dtype=np.float32)
    normalized = (aggregate - stats.aggregate_mean) / stats.aggregate_std
    windows = np.stack([normalized[start : start + window_size] for start in chosen])
    result = torch.as_tensor(windows, dtype=torch.float32)
    mask = torch.ones(context_k, dtype=torch.bool)
    if count < context_k:
        padding = torch.zeros(context_k - count, window_size, dtype=torch.float32)
        result = torch.cat((result, padding), dim=0)
        mask[count:] = False
    timestamps = aggregate_block["time"].iloc[chosen].to_numpy()
    return result, mask, timestamps


class QueryWindowDataset(Dataset):
    """Lazy point-target windows that never cross repaired segment boundaries."""

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        house: str,
        stats: NormalisationStats,
        window_size: int,
        appliance: str = "washing_machine",
        require_target: bool = True,
    ):
        self.house = house
        self.window_size = int(window_size)
        self.center_offset = self.window_size // 2
        self.frame = frame.reset_index(drop=True)
        self.aggregate_raw = self.frame["aggregate"].to_numpy(dtype=np.float32)
        self.aggregate = (self.aggregate_raw - stats.aggregate_mean) / stats.aggregate_std
        self.target_raw = None
        self.target = None
        if require_target:
            if appliance not in self.frame.columns:
                raise ValueError(f"Query training/validation frame lacks {appliance}.")
            self.target_raw = self.frame[appliance].to_numpy(dtype=np.float32)
            self.target = (self.target_raw - stats.appliance_mean) / stats.appliance_std
        self.starts = _valid_window_starts(self.frame, self.window_size)
        if len(self.starts) == 0:
            raise ValueError(f"{house} query block has no complete windows.")
        self.center_indices = self.starts + self.center_offset
        self.timestamps = self.frame["time"].iloc[self.center_indices].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int):
        start = int(self.starts[index])
        query = torch.as_tensor(
            self.aggregate[start : start + self.window_size], dtype=torch.float32
        )
        if self.target is None:
            return query
        target = torch.tensor(self.target[start + self.center_offset], dtype=torch.float32)
        return query, target


class MultiHouseQueryDataset(Dataset):
    def __init__(self, datasets: Sequence[QueryWindowDataset]):
        if not datasets:
            raise ValueError("At least one household query dataset is required.")
        self.datasets = list(datasets)
        self.offsets = np.cumsum([0, *[len(dataset) for dataset in self.datasets]])
        self.house_ranges = {
            dataset.house: np.arange(self.offsets[i], self.offsets[i + 1], dtype=np.int64)
            for i, dataset in enumerate(self.datasets)
        }

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def __getitem__(self, index: int):
        dataset_index = int(np.searchsorted(self.offsets[1:], index, side="right"))
        local_index = int(index - self.offsets[dataset_index])
        query, target = self.datasets[dataset_index][local_index]
        return query, target, dataset_index


class HouseholdBalancedBatchSampler(Sampler[list[int]]):
    """Equal examples per household in every batch, reproducibly cycled."""

    def __init__(
        self,
        dataset: MultiHouseQueryDataset,
        *,
        batch_size: int,
        seed: int,
        epoch_size: int | None = None,
    ):
        self.dataset = dataset
        self.num_houses = len(dataset.datasets)
        if batch_size < self.num_houses or batch_size % self.num_houses:
            raise ValueError("Balanced batch_size must be divisible by the number of households.")
        self.per_house = batch_size // self.num_houses
        self.seed = int(seed)
        max_house = max(len(indices) for indices in dataset.house_ranges.values())
        default_size = max_house * self.num_houses
        self.epoch_size = int(epoch_size or default_size)
        self.epoch_size -= self.epoch_size % batch_size
        self.batch_size = batch_size
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.epoch_size // self.batch_size

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        pools: list[np.ndarray] = []
        cursors: list[int] = []
        for indices in self.dataset.house_ranges.values():
            pools.append(rng.permutation(indices))
            cursors.append(0)
        for _ in range(len(self)):
            batch: list[int] = []
            for house_index, original in enumerate(self.dataset.house_ranges.values()):
                if cursors[house_index] + self.per_house > len(pools[house_index]):
                    pools[house_index] = rng.permutation(original)
                    cursors[house_index] = 0
                start = cursors[house_index]
                batch.extend(pools[house_index][start : start + self.per_house].tolist())
                cursors[house_index] += self.per_house
            rng.shuffle(batch)
            yield batch


class HouseholdUniformBatchSampler(Sampler[list[int]]):
    """Sample one household uniformly per batch, then sample only its queries."""

    def __init__(
        self,
        dataset: MultiHouseQueryDataset,
        *,
        batch_size: int,
        seed: int,
        epoch_size: int | None = None,
    ):
        self.dataset = dataset
        self.house_ranges = list(dataset.house_ranges.values())
        self.num_houses = len(self.house_ranges)
        self.batch_size = int(batch_size)
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive.")
        self.seed = int(seed)
        default_size = max(len(indices) for indices in self.house_ranges) * self.num_houses
        self.epoch_size = int(epoch_size or default_size)
        self.epoch_size -= self.epoch_size % self.batch_size
        if self.epoch_size < self.batch_size:
            raise ValueError("epoch_size must contain at least one complete batch.")
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.epoch_size // self.batch_size

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        pools = [rng.permutation(indices) for indices in self.house_ranges]
        cursors = [0 for _ in self.house_ranges]
        for _ in range(len(self)):
            house_index = int(rng.integers(self.num_houses))
            original = self.house_ranges[house_index]
            if cursors[house_index] + self.batch_size > len(pools[house_index]):
                pools[house_index] = rng.permutation(original)
                cursors[house_index] = 0
            start = cursors[house_index]
            batch = pools[house_index][start : start + self.batch_size].tolist()
            cursors[house_index] += self.batch_size
            yield batch


def timestamp_fingerprint(timestamps: Sequence[object]) -> str:
    import hashlib

    values = pd.to_datetime(pd.Series(timestamps)).astype("int64").to_numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()


def assert_common_evaluation_timestamps(
    method_timestamps: dict[str, Sequence[object]],
) -> str:
    if not method_timestamps:
        raise ValueError("No evaluation timestamps were provided.")
    fingerprints = {
        method: timestamp_fingerprint(timestamps)
        for method, timestamps in method_timestamps.items()
    }
    if len(set(fingerprints.values())) != 1:
        raise RuntimeError(f"Evaluation timestamp mismatch: {fingerprints}")
    return next(iter(fingerprints.values()))


def resolve_block_specs(config: dict, repo_root: Path) -> dict[str, dict[str, BlockSpec]]:
    """Expand manifest-fixed 14-day starts into adjacent A/B seven-day blocks."""
    data_dir = repo_root / config["data"]["csv_dir"]
    result: dict[str, dict[str, BlockSpec]] = {}
    for house, start_value in config["fixed_14_day_starts"].items():
        start = pd.Timestamp(start_value)
        middle = start + pd.Timedelta(days=7)
        end = start + pd.Timedelta(days=14)
        csv_path = data_dir / config["data"]["filename_template"].format(
            appliance=config["appliance"], house=house
        )
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        result[house] = {
            "A": BlockSpec(house, "A", start.isoformat(), middle.isoformat(), str(csv_path)),
            "B": BlockSpec(house, "B", middle.isoformat(), end.isoformat(), str(csv_path)),
        }
    return result
