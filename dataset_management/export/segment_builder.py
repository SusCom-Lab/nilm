from __future__ import annotations

import pandas as pd


class SegmentBuilder:
    def build_segments(self, df: pd.DataFrame, resample_interval: str) -> pd.DataFrame:
        if df.empty:
            result = df.copy()
            result["segment_id"] = pd.Series(dtype="int64")
            return result

        result = df.copy()
        result["time"] = pd.to_datetime(result["time"], errors="coerce")
        result = result.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

        aggregate_usable = result["aggregate_is_valid"].infer_objects(copy=False).fillna(False).astype(bool)
        appliance_usable = result["appliance_is_valid"].infer_objects(copy=False).fillna(False).astype(bool)
        usable = aggregate_usable & appliance_usable
        delta = result["time"].diff()
        expected = pd.Timedelta(resample_interval)
        previous_usable = usable.shift(fill_value=False)

        segment_start = usable & ((~previous_usable) | (delta != expected))
        segment_id = segment_start.cumsum() - 1

        result["segment_id"] = segment_id.astype("int64")
        return result.loc[usable].reset_index(drop=True)
