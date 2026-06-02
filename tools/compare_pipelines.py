from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from model_pipeline.model_registry import create_model
from model_pipeline.test_model import Evaluator
from model_pipeline.train_model import Trainer, set_random_seed


class LegacySlidingWindowDataset(Dataset):
    def __init__(
        self,
        file_dirs,
        window_size,
        *,
        split_ratio=None,
        split_mode=None,
        target_mode="point",
        output_size=None,
        output_offset=None,
    ):
        self.window_size = int(window_size)
        self.target_mode = target_mode
        self.output_size = int(output_size or (1 if target_mode == "point" else self.window_size))
        self.output_offset = int(output_offset if output_offset is not None else (self.window_size // 2))
        self.normalisation_params = {}
        all_dfs = []

        for file in file_dirs:
            df = pd.read_csv(file)
            house_params = {
                "aggregate_mean": float(df.iloc[:, 1].mean()),
                "aggregate_std": _safe_std(df.iloc[:, 1].std()),
                "appliance_mean": float(df.iloc[:, 2].mean()),
                "appliance_std": _safe_std(df.iloc[:, 2].std()),
            }
            df = df.copy()
            df.iloc[:, 1] = (df.iloc[:, 1] - house_params["aggregate_mean"]) / house_params["aggregate_std"]
            df.iloc[:, 2] = (df.iloc[:, 2] - house_params["appliance_mean"]) / house_params["appliance_std"]
            self.normalisation_params[file] = house_params
            all_dfs.append(df)

        merged_df = pd.concat(all_dfs, ignore_index=True)
        if split_ratio is not None and split_mode is not None:
            split_index = int(len(merged_df) * split_ratio)
            if split_mode == "train":
                merged_df = merged_df.iloc[:split_index]
            elif split_mode == "val":
                merged_df = merged_df.iloc[split_index:]

        self.inputs = torch.tensor(merged_df.iloc[:, 1].to_numpy(), dtype=torch.float32)
        self.outputs = torch.tensor(merged_df.iloc[:, 2].to_numpy(), dtype=torch.float32)

    def __len__(self):
        last_required_index = max(self.window_size, self.output_offset + self.output_size)
        return max(0, len(self.inputs) - last_required_index + 1)

    def __getitem__(self, idx):
        inputs = self.inputs[idx:idx + self.window_size]
        if self.target_mode == "point":
            target = self.outputs[idx + self.output_offset]
        else:
            target = self.outputs[idx + self.output_offset:idx + self.output_offset + self.output_size]
        return inputs, target

    def getNormalisationParams(self, file_dir):
        return self.normalisation_params[file_dir]


@dataclass
class RunResult:
    label: str
    best_val_loss: float
    mae: float
    sae: float


def _safe_std(value):
    value = float(value)
    if not np.isfinite(value) or value == 0.0:
        return 1.0
    return value


def _build_legacy_loaders(csv_paths, model, batch_size, val_ratio, seed):
    split_ratio = 1 - val_ratio
    train_dataset = LegacySlidingWindowDataset(
        csv_paths,
        model.get_window_size(),
        split_ratio=split_ratio,
        split_mode="train",
        target_mode=model.get_target_type(),
        output_size=model.get_output_size(),
        output_offset=model.get_output_offset(),
    )
    validation_dataset = LegacySlidingWindowDataset(
        csv_paths,
        model.get_window_size(),
        split_ratio=split_ratio,
        split_mode="val",
        target_mode=model.get_target_type(),
        output_size=model.get_output_size(),
        output_offset=model.get_output_offset(),
    )

    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, validation_loader


def _train_and_eval(label, trainer, test_csv_path, dataset_name, epochs, normalisation_params=None):
    trainer.trainModel(num_epochs=epochs)
    checkpoint_path = trainer._save_checkpoint()
    evaluator = Evaluator(
        model_state_dir=checkpoint_path,
        test_csv_dir=test_csv_path,
        dataset=dataset_name,
        normalisation_params=normalisation_params,
    )
    mae, sae, _ = evaluator.evaluate()
    return RunResult(label=label, best_val_loss=trainer.best_val_loss, mae=mae, sae=sae)


def _print_result(result):
    print(
        f"{result.label}: best_val_loss={result.best_val_loss:.6f}, "
        f"MAE={result.mae:.6f}, SAE={result.sae:.6f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare legacy and current data pipelines.")
    parser.add_argument("--dataset", default="ukdale")
    parser.add_argument("--appliance", default="dishwasher")
    parser.add_argument("--model", default="Seq2Point")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--window-size", type=int, default=599)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-csv", nargs="+", required=True)
    parser.add_argument("--test-csv", required=True)
    args = parser.parse_args()

    set_random_seed(args.seed)
    base_model = create_model(args.model, window_size=args.window_size)

    current_trainer = Trainer(
        model=copy.deepcopy(base_model),
        train_csv_dirs=args.train_csv,
        appliance=args.appliance,
        dataset=args.dataset,
        window_length=args.window_size,
        val_ratio=args.val_ratio,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    current_result = _train_and_eval("current", current_trainer, args.test_csv, args.dataset, args.epochs)

    legacy_model = copy.deepcopy(base_model)
    legacy_train_loader, legacy_val_loader = _build_legacy_loaders(
        args.train_csv,
        legacy_model,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    legacy_trainer = Trainer(
        model=legacy_model,
        train_loader=legacy_train_loader,
        validation_loader=legacy_val_loader,
        appliance=args.appliance,
        dataset=args.dataset,
        window_length=args.window_size,
        batch_size=args.batch_size,
        seed=args.seed,
        normalisation_stats=legacy_train_loader.dataset.getNormalisationParams(args.train_csv[0]),
    )
    legacy_norm = legacy_val_loader.dataset.getNormalisationParams(args.train_csv[0])
    legacy_result = _train_and_eval(
        "legacy",
        legacy_trainer,
        args.test_csv,
        args.dataset,
        args.epochs,
        normalisation_params=legacy_norm,
    )

    _print_result(current_result)
    _print_result(legacy_result)
