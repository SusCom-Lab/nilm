"""
Official-style experiment for house-invariant StateAwareSeq2Point.

This script trains on one or more source-house CSV files. Each source file is
treated as one domain label for the adversarial house/domain head.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import ConcatDataset, DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model_pipeline.data_feeder import SlidingWindowDataset
from model_pipeline.models.seq2point.seq2point_invariant_state_aware import InvariantStateAwareSeq2Point
from model_pipeline.models.seq2point.seq2point_state_aware import power_to_status
from model_pipeline.test_model import Evaluator
from model_pipeline.train_model import set_random_seed


STATUS_RULES_FILE = ROOT / "model_pipeline" / "appliance_status_rules.json"


class HouseLabelDataset(Dataset):
    def __init__(self, dataset: Dataset, house_id: int):
        self.dataset = dataset
        self.house_id = int(house_id)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        inputs, targets = self.dataset[idx]
        return inputs, targets, torch.tensor(self.house_id, dtype=torch.long)


def load_status_threshold(appliance: str) -> float:
    with STATUS_RULES_FILE.open("r", encoding="utf-8") as handle:
        rules = json.load(handle)
    key = appliance.replace(" ", "_").lower()
    return float(rules[key]["status_threshold"])


def normalise_threshold(raw_threshold: float, stats: dict[str, float]) -> float:
    return (raw_threshold - stats["appliance_mean"]) / stats["appliance_std"]


def estimate_pos_weight(loader, threshold: float) -> float:
    positives = 0
    total = 0
    for _inputs, targets, _house_ids in loader:
        labels = power_to_status(targets, threshold)
        positives += int(labels.sum().item())
        total += int(labels.numel())
    return float(max(0, total - positives) / max(1, positives))


def house_label_from_path(path: str, fallback: int) -> str:
    match = re.search(r"_H(\d+)\.csv$", Path(path).name)
    if match:
        return f"H{match.group(1)}"
    return f"domain{fallback}"


def build_domain_loaders(args):
    train_split_ratio = 1.0 - args.val_ratio
    stats_dataset = SlidingWindowDataset(
        args.train_csvs,
        args.window_size,
        crop=args.crop,
        split_ratio=train_split_ratio,
        split_mode="train",
        target_mode="point",
        output_size=1,
        output_offset=args.window_size // 2,
    )
    stats = stats_dataset.get_normalisation_stats()

    train_parts = []
    val_parts = []
    domain_labels = []
    for domain_id, csv_path in enumerate(args.train_csvs):
        domain_labels.append(house_label_from_path(csv_path, domain_id))
        train_dataset = SlidingWindowDataset(
            [csv_path],
            args.window_size,
            crop=args.crop,
            split_ratio=train_split_ratio,
            split_mode="train",
            target_mode="point",
            output_size=1,
            output_offset=args.window_size // 2,
            normalisation_stats=stats,
        )
        val_dataset = SlidingWindowDataset(
            [csv_path],
            args.window_size,
            crop=args.crop,
            split_ratio=train_split_ratio,
            split_mode="val",
            target_mode="point",
            output_size=1,
            output_offset=args.window_size // 2,
            normalisation_stats=stats,
        )
        if len(train_dataset) > 0:
            train_parts.append(HouseLabelDataset(train_dataset, domain_id))
        if len(val_dataset) > 0:
            val_parts.append(HouseLabelDataset(val_dataset, domain_id))

    if not train_parts:
        raise ValueError("No training windows were produced.")
    if not val_parts:
        raise ValueError("No validation windows were produced.")

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        ConcatDataset(train_parts),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(
        ConcatDataset(val_parts),
        batch_size=args.batch_size,
        shuffle=False,
    )
    return train_loader, val_loader, stats, domain_labels


def train_model(args, model, train_loader, val_loader, stats):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, threshold=1e-4)
    best_selection_loss = float("inf")
    patience_counter = 0
    best_path = Path(args.model_save_dir) / f"{args.appliance}_{args.dataset}_InvariantStateAwareSeq2Point.pth"

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_power_loss = 0.0
        train_state_loss = 0.0
        train_house_loss = 0.0
        for inputs, targets, house_ids in train_loader:
            inputs = inputs.to(device)
            targets = model.prepare_targets(targets.to(device))
            house_ids = house_ids.to(device)
            optimizer.zero_grad()
            loss, _outputs, details = model.compute_loss(inputs, targets, house_ids, criterion)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())
            train_power_loss += float(details["power_loss"].item())
            train_state_loss += float(details["state_loss"].item())
            train_house_loss += float(details["house_loss"].item())

        denom = max(1, len(train_loader))
        train_loss /= denom
        train_power_loss /= denom
        train_state_loss /= denom
        train_house_loss /= denom

        model.eval()
        val_loss = 0.0
        selection_loss = 0.0
        with torch.no_grad():
            for inputs, targets, house_ids in val_loader:
                inputs = inputs.to(device)
                targets = model.prepare_targets(targets.to(device))
                house_ids = house_ids.to(device)
                loss, _outputs, _details = model.compute_loss(inputs, targets, house_ids, criterion)
                val_loss += float(loss.item())
                selection_loss += float(model.compute_selection_loss(inputs, targets, criterion).item())
        val_loss /= max(1, len(val_loader))
        selection_loss /= max(1, len(val_loader))
        scheduler.step(selection_loss)

        print(
            f"Epoch {epoch + 1}/{args.epochs}, Train Loss: {train_loss:.6f}, "
            f"Train Power: {train_power_loss:.6f}, Train State: {train_state_loss:.6f}, "
            f"Train House: {train_house_loss:.6f}, Val Loss: {val_loss:.6f}, "
            f"Selection Loss: {selection_loss:.6f}"
        )

        if selection_loss < best_selection_loss - args.min_delta:
            best_selection_loss = selection_loss
            patience_counter = 0
            save_checkpoint(best_path, model, args, stats)
            print(f"Validation improved. Checkpoint saved to {best_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

    return best_path


def save_checkpoint(path: Path, model, args, stats):
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_name": model.display_name,
        "model_key": getattr(model, "_registry_key", model.display_name),
        "window_length": model.get_window_size(),
        "output_size": model.get_output_size(),
        "output_offset": model.get_output_offset(),
        "target_type": model.get_target_type(),
        "appliance": args.appliance,
        "dataset": args.dataset,
        "init_kwargs": model.get_init_kwargs(),
        "model_state": model.export_state(),
        "model_state_dict": model.export_state(),
        "normalisation_stats": stats,
        "domain_labels": args.domain_labels,
    }
    torch.save(checkpoint, path)


def evaluate_checkpoint(checkpoint: Path, test_csv: str, args) -> None:
    result_dir = Path(args.result_dir)
    if len(args.test_csvs) > 1:
        suffix = house_label_from_path(test_csv, 0)
        result_dir = result_dir / suffix
    evaluator = Evaluator(
        model_state_dir=str(checkpoint),
        test_csv_dir=test_csv,
        dataset=args.dataset,
        result_dir=str(result_dir),
        batch_size=args.batch_size,
    )
    evaluator.testModel()
    evaluator.saveMetrics()
    results_path = result_dir / f"{evaluator.appliance_name_formatted}_{evaluator.model_name}_results.csv"
    evaluator.getResults().to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--appliance", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--train-csvs", nargs="+", required=True)
    parser.add_argument("--test-csvs", nargs="+", required=True)
    parser.add_argument("--crop", type=int, default=300000)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--window-size", type=int, default=599)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--state-loss-weight", type=float, default=0.05)
    parser.add_argument("--house-loss-weight", type=float, default=0.003)
    parser.add_argument("--grl-lambda", type=float, default=1.0)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--model-save-dir", required=True)
    parser.add_argument("--result-dir", required=True)
    args = parser.parse_args()

    Path(args.model_save_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
    set_random_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    train_loader, val_loader, stats, domain_labels = build_domain_loaders(args)
    args.domain_labels = domain_labels
    raw_threshold = load_status_threshold(args.appliance)
    state_threshold = normalise_threshold(raw_threshold, stats)
    state_pos_weight = estimate_pos_weight(train_loader, state_threshold)
    print(
        f"Invariant settings: raw_threshold={raw_threshold:.3f}, "
        f"normalised_threshold={state_threshold:.6f}, pos_weight={state_pos_weight:.3f}, "
        f"domains={domain_labels}, house_loss_weight={args.house_loss_weight}"
    )

    set_random_seed(args.seed)
    model = InvariantStateAwareSeq2Point(
        window_size=args.window_size,
        hidden_dim=args.hidden_dim,
        state_threshold=state_threshold,
        state_loss_weight=args.state_loss_weight,
        state_pos_weight=state_pos_weight,
        num_domains=len(domain_labels),
        house_loss_weight=args.house_loss_weight,
        grl_lambda=args.grl_lambda,
    )
    checkpoint = train_model(args, model, train_loader, val_loader, stats)
    for test_csv in args.test_csvs:
        evaluate_checkpoint(checkpoint, test_csv, args)


if __name__ == "__main__":
    main()
