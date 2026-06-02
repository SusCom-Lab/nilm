from __future__ import annotations

import pandas as pd


def identify_signal_column(df: pd.DataFrame) -> str:
    excluded = {"time", "is_original", "is_filled_short", "is_valid_point"}
    candidates = [col for col in df.columns if col not in excluded]
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one signal column, found {candidates}.")
    return candidates[0]


def deduplicate_by_time(df: pd.DataFrame, signal_column: str) -> pd.DataFrame:
    return df.groupby("time", as_index=False)[signal_column].mean()


def resample_signal(df: pd.DataFrame, signal_column: str, interval: str) -> pd.DataFrame:
    resampled = df.set_index("time").resample(interval).mean()
    resampled.index.name = "time"
    return resampled.reset_index()


def forward_fill_limited(series: pd.Series, limit: int) -> pd.Series:
    if limit <= 0:
        return series.copy()
    return series.ffill(limit=limit)


def clip_power_values(series: pd.Series, min_threshold: float, max_power_clip: float) -> pd.Series:
    clipped = pd.to_numeric(series, errors="coerce").clip(lower=0, upper=max_power_clip)
    clipped = clipped.mask(clipped < min_threshold, 0.0)
    return clipped
