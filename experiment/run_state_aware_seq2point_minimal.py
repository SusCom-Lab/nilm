"""
Minimal feasibility experiment for State-aware Seq2Point.

Runs a small window-level comparison between the original Seq2Point and the
state-aware variant on one appliance CSV. The goal is to check whether the
state auxiliary head can reduce false positives without hurting power error.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model_pipeline.models.seq2point.seq2point import Seq2Point
from model_pipeline.models.seq2point.seq2point_state_aware import StateAwareSeq2Point, power_to_status
from model_pipeline.data_feeder import SlidingWindowDataset
from model_pipeline.train_model import build_windowed_loaders, set_random_seed


STATUS_RULES_FILE = ROOT / "model_pipeline" / "appliance_status_rules.json"


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
    for _, targets in loader:
        labels = power_to_status(targets, threshold)
        positives += int(labels.sum().item())
        total += int(labels.numel())
    negatives = max(0, total - positives)
    return float(negatives / max(1, positives))


def train_one_model(model, train_loader, *, epochs: int, lr: float, device: str) -> None:
    model.to(device)
    model.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    for epoch in range(epochs):
        total_loss = 0.0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = model.prepare_targets(targets.to(device))
            optimizer.zero_grad()
            loss_hook = getattr(model, "compute_loss", None)
            if callable(loss_hook):
                loss, _ = loss_hook(inputs, targets, criterion=criterion)
            else:
                outputs = model.prepare_outputs(model(inputs))
                loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        print(f"{model.display_name} epoch {epoch + 1}/{epochs} loss={total_loss / max(1, len(train_loader)):.6f}")


def evaluate_window_metrics(model, loader, *, stats: dict[str, float], raw_threshold: float, device: str) -> dict[str, float]:
    model.to(device)
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for inputs, batch_targets in loader:
            inputs = inputs.to(device)
            outputs = model.prepare_outputs(model(inputs)).detach().cpu().numpy()
            predictions.append(outputs.reshape(-1))
            targets.append(model.prepare_targets(batch_targets).detach().cpu().numpy().reshape(-1))

    if not predictions:
        raise ValueError("Validation loader produced no windows. Increase --crop or reduce --window-size.")

    pred_norm = np.concatenate(predictions)
    target_norm = np.concatenate(targets)
    pred = np.clip((pred_norm * stats["appliance_std"]) + stats["appliance_mean"], 0.0, None)
    target = np.clip((target_norm * stats["appliance_std"]) + stats["appliance_mean"], 0.0, None)

    true_state = target >= raw_threshold
    pred_state = pred >= raw_threshold
    tp = int(np.logical_and(pred_state, true_state).sum())
    fp = int(np.logical_and(pred_state, ~true_state).sum())
    tn = int(np.logical_and(~pred_state, ~true_state).sum())
    fn = int(np.logical_and(~pred_state, true_state).sum())

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    fpr = fp / max(1, fp + tn)

    return {
        "MAE": float(np.mean(np.abs(pred - target))),
        "RMSE": float(np.sqrt(np.mean((pred - target) ** 2))),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
        "FPR": float(fpr),
        "TP": float(tp),
        "FP": float(fp),
        "TN": float(tn),
        "FN": float(fn),
        "positive_ratio": float(true_state.mean()),
    }


def build_test_loader(
    csv_path: str,
    *,
    window_size: int,
    batch_size: int,
    crop: int | None,
    stats: dict[str, float],
) -> DataLoader:
    dataset = SlidingWindowDataset(
        [csv_path],
        window_size,
        crop=crop,
        target_mode="point",
        output_size=1,
        output_offset=window_size // 2,
        normalisation_stats=stats,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--appliance", default="kettle")
    parser.add_argument("--csv", default=str(ROOT / "dataset" / "UKDALE_dataset" / "kettle_H1.csv"))
    parser.add_argument("--test-csv", default=None)
    parser.add_argument("--crop", type=int, default=100000)
    parser.add_argument("--test-crop", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--window-size", type=int, default=599)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--state-loss-weight", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=str(ROOT / "experiment" / "state_aware_seq2point_minimal_results.csv"))
    args = parser.parse_args()

    set_random_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    raw_threshold = load_status_threshold(args.appliance)

    train_loader, val_loader, stats = build_windowed_loaders(
        [args.csv],
        window_size=args.window_size,
        target_mode="point",
        output_size=1,
        output_offset=args.window_size // 2,
        crop=args.crop,
        batch_size=args.batch_size,
        val_ratio=0.2,
        seed=args.seed,
    )
    test_loader = None
    if args.test_csv:
        test_loader = build_test_loader(
            args.test_csv,
            window_size=args.window_size,
            batch_size=args.batch_size,
            crop=args.test_crop,
            stats=stats,
        )
    state_threshold = normalise_threshold(raw_threshold, stats)
    state_pos_weight = estimate_pos_weight(train_loader, state_threshold)
    print(
        f"appliance={args.appliance} raw_threshold={raw_threshold:.3f} "
        f"normalised_threshold={state_threshold:.6f} pos_weight={state_pos_weight:.3f}"
    )

    models = [
        Seq2Point(window_size=args.window_size, hidden_dim=args.hidden_dim),
        StateAwareSeq2Point(
            window_size=args.window_size,
            hidden_dim=args.hidden_dim,
            state_threshold=state_threshold,
            state_loss_weight=args.state_loss_weight,
            state_pos_weight=state_pos_weight,
        ),
    ]

    rows = []
    for model in models:
        train_one_model(model, train_loader, epochs=args.epochs, lr=args.lr, device=device)
        val_metrics = evaluate_window_metrics(model, val_loader, stats=stats, raw_threshold=raw_threshold, device=device)
        val_row = {"split": "validation", "model": model.display_name, "appliance": args.appliance, **val_metrics}
        rows.append(val_row)
        print(val_row)
        if test_loader is not None:
            test_metrics = evaluate_window_metrics(
                model,
                test_loader,
                stats=stats,
                raw_threshold=raw_threshold,
                device=device,
            )
            test_row = {"split": "test", "model": model.display_name, "appliance": args.appliance, **test_metrics}
            rows.append(test_row)
            print(test_row)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
