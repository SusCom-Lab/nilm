from __future__ import annotations

import os
from typing import List

import pandas as pd


class StandardH5Adapter:
    """
    Adapter for datasets that already follow the standard NILM HDF5 layout:

        /House_<id>/aggregate
        /House_<id>/<appliance_name>
    """

    def list_houses(self, h5_path: str) -> List[str]:
        with pd.HDFStore(h5_path, mode="r") as store:
            houses = {
                key.strip("/").split("/")[0]
                for key in store.keys()
                if key.strip("/").count("/") >= 1
            }
        return sorted(houses)

    def list_signals(self, h5_path: str, house_id: str) -> List[str]:
        prefix = f"/{house_id}/"
        with pd.HDFStore(h5_path, mode="r") as store:
            signals = [
                key[len(prefix):]
                for key in store.keys()
                if key.startswith(prefix) and key.count("/") == 2
            ]
        return sorted(signals)

    def load_signal(self, h5_path: str, house_id: str, signal_name: str) -> pd.DataFrame:
        key = f"/{house_id}/{signal_name}"
        df = pd.read_hdf(h5_path, key=key)
        if "time" not in df.columns:
            raise ValueError(f"{key} is missing required 'time' column.")
        if signal_name not in df.columns:
            power_columns = [col for col in df.columns if col != "time"]
            if len(power_columns) != 1:
                raise ValueError(f"{key} must contain exactly one power column.")
            df = df.rename(columns={power_columns[0]: signal_name})
        return df[["time", signal_name]].copy()

    def export_to_separated_files(self, h5_path: str, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)

        for house_name in self.list_houses(h5_path):
            house_number = house_name.split("_", 1)[1]
            house_dir = os.path.join(output_dir, house_name)
            os.makedirs(house_dir, exist_ok=True)

            for signal_name in self.list_signals(h5_path, house_name):
                signal_df = self.load_signal(h5_path, house_name, signal_name)
                output_file = os.path.join(house_dir, f"{signal_name}_H{house_number}.h5")
                signal_df.to_hdf(output_file, key="dataset", mode="w", format="table")
