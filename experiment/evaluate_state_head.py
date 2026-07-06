"""
Evaluate the StateAwareSeq2Point state head directly.

This is separate from the normal Evaluator because the normal pipeline derives
on/off status from predicted power. Here we measure the auxiliary state branch
itself against threshold-derived appliance states.
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
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model_pipeline.data_feeder import SlidingWindowDataset
from model_pipeline.model_registry import instantiate_from_checkpoint, load_checkpoint


STATUS_RULES_FILE = ROOT / "model_pipeline" / "appliance_status_rules.json"


def load_status_threshold(appliance: str) -> float:
    with STATUS_RULES_FILE.open("r", encoding="utf-8") as handle:
        rules = json.load(handle)
    key = appliance.replace(" ", "_").lower()
    return float(rules[key]["status_threshold"])


def binary_metrics(y_true: np.ndarray, y_score: np.ndarray, decision_threshold: float) -> dict[str, float]:
    y_pred = y_score >= decision_threshold
    y_true_bool = y_true.astype(bool)
    tp = int(np.logical_and(y_pred, y_true_bool).sum())
    fp = int(np.logical_and(y_pred, ~y_true_bool).sum())
    tn = int(np.logical_and(~y_pred, ~y_true_bool).sum())
    fn = int(np.logical_and(~y_pred, y_true_bool).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    fpr = fp / max(1, fp + tn)
    pr_auc = float("nan")
    if len(np.unique(y_true_bool)) > 1:
        pr_auc = float(average_precision_score(y_true_bool, y_score))
    return {
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
        "FPR": float(fpr),
        "PR-AUC": pr_auc,
        "TP": float(tp),
        "FP": float(fp),
        "TN": float(tn),
        "FN": float(fn),
        "positive_ratio": float(y_true_bool.mean()),
    }


def evaluate_state_head(args) -> dict[str, float | str]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    model = instantiate_from_checkpoint(checkpoint, map_location=device)
    if not hasattr(model, "state_head") or not hasattr(model, "encode"):
        raise TypeError("Checkpoint model does not expose encode() and state_head.")
    model.to(device)
    model.eval()

    appliance = args.appliance or checkpoint.get("appliance", "appliance")
    raw_threshold = load_status_threshold(appliance)
    stats = checkpoint.get("normalisation_stats")
    if stats is None:
        raise ValueError("Checkpoint is missing normalisation_stats.")

    dataset = SlidingWindowDataset(
        [args.test_csv],
        model.get_window_size(),
        crop=args.crop,
        target_mode=model.get_target_type(),
        output_size=model.get_output_size(),
        output_offset=model.get_output_offset(),
        normalisation_stats=stats,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    scores = []
    targets = []
    with torch.no_grad():
        for inputs, batch_targets in loader:
            inputs = inputs.to(device)
            logits = model.state_head(model.encode(inputs)).reshape(-1)
            scores.append(torch.sigmoid(logits).detach().cpu().numpy())
            targets.append(model.prepare_targets(batch_targets).detach().cpu().numpy().reshape(-1))

    if not scores:
        raise ValueError("Test loader produced no windows.")

    y_score = np.concatenate(scores).astype(np.float32)
    target_norm = np.concatenate(targets).astype(np.float32)
    target_raw = (target_norm * float(stats["appliance_std"])) + float(stats["appliance_mean"])
    y_true = target_raw >= raw_threshold
    metrics = binary_metrics(y_true.astype(np.int8), y_score, args.decision_threshold)
    return {
        "checkpoint": args.checkpoint,
        "test_csv": args.test_csv,
        "appliance": appliance,
        "raw_threshold": raw_threshold,
        "decision_threshold": args.decision_threshold,
        "num_windows": float(len(y_score)),
        **metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--appliance", default=None)
    parser.add_argument("--crop", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--decision-threshold", type=float, default=0.5)
    parser.add_argument("--output", default=str(ROOT / "experiment" / "state_head_metrics.csv"))
    args = parser.parse_args()

    row = evaluate_state_head(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output.exists()
    with output.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(row)
    print(f"Saved state-head metrics to {output}")


if __name__ == "__main__":
    main()
