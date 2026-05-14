import torch
from torch.utils.data import Dataset
import pandas as pd

class SlidingWindowDataset(Dataset):
    def __init__(self, file_dirs, window_size, crop=None, split_ratio=None, split_mode=None):
        self.window_size = window_size
        self.offset = (window_size // 2) - 1
        self.data = []
        self.normalisation_params = {}

        all_dfs = []
        for file in file_dirs:
            house_params = {}
            df = pd.read_csv(file)
            if crop:
                df = df.head(crop)

            house_params["aggregate_mean"] = df.iloc[:, 1].mean()
            house_params["aggregate_std"] = df.iloc[:, 1].std()
            house_params["appliance_mean"] = df.iloc[:, 2].mean()
            house_params["appliance_std"] = df.iloc[:, 2].std()

            df.iloc[:, 1] = (df.iloc[:, 1] - house_params["aggregate_mean"]) / house_params["aggregate_std"]
            df.iloc[:, 2] = (df.iloc[:, 2] - house_params["appliance_mean"]) / house_params["appliance_std"]

            self.normalisation_params[file] = house_params
            all_dfs.append(df)
        
        merged_df = pd.concat(all_dfs, ignore_index=True)

        if split_ratio is not None and split_mode is not None:
            total_rows = len(merged_df)
            split_index = int(total_rows * split_ratio)
            
            if split_mode == 'train':
                merged_df = merged_df.iloc[:split_index]
            elif split_mode == 'val':
                merged_df = merged_df.iloc[split_index:]

        inputs = torch.tensor(merged_df.iloc[:, 1].values, dtype=torch.float32)
        outputs = torch.tensor(merged_df.iloc[:, 2].values, dtype=torch.float32)
        self.data.append((inputs, outputs))

    def __len__(self):
        return sum(len(inputs) - self.window_size + 1 for inputs, _ in self.data)

    def getNormalisationParams(self, file_dir):
        return self.normalisation_params[file_dir]

    def __getitem__(self, idx):
        for inputs, outputs in self.data:
            num_windows = len(inputs) - self.window_size + 1
            if idx < num_windows:
                start_idx = idx
                end_idx = idx + self.window_size
                return inputs[start_idx:end_idx], outputs[start_idx + self.offset]
            idx -= num_windows

        raise IndexError("Index out of range")
