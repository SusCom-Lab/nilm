"""
Evaluate inference-time state gating for StateAwareSeq2Point.

The normal evaluator derives on/off status only from predicted power. This
script keeps the trained power prediction fixed and tests whether the auxiliary
state head can filter false positives at inference time:

    gated_on = power_on AND state_probability >= threshold
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model_pipeline.data_feeder import SlidingWindowDataset, reconstruct_series_from_windows
from model_pipeline.model_registry import instantiate_from_checkpoint, load_checkpoint
from model_pipeline.test_model import _get_status_rule, _power_to_status


DEFAULT_APPLIANCES = ("dishwasher", "washing_machine", "fridge", "kettle", "microwave")
DEFAULT_THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    tp = int(np.logical_and(y_pred, y_true).sum())
    fp = int(np.logical_and(y_pred, ~y_true).sum())
    tn = int(np.logical_and(~y_pred, ~y_true).sum())
    fn = int(np.logical_and(~y_pred, y_true).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    fpr = fp / max(1, fp + tn)
    pr_auc = float("nan")
    if len(np.unique(y_true)) > 1:
        pr_auc = float(average_precision_score(y_true, y_score))
    return {
        "Precision": float(precision),
        "Recall": float(recall),
        "F1-score": float(f1),
        "FPR": float(fpr),
        "PR-AUC": pr_auc,
        "TP": float(tp),
        "FP": float(fp),
        "TN": float(tn),
        "FN": float(fn),
        "status_positive_ratio": float(y_true.mean()),
    }


def reconstruct_state_probabilities(checkpoint_path: Path, test_csv: Path, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = load_checkpoint(str(checkpoint_path), map_location=device)
    model = instantiate_from_checkpoint(checkpoint, map_location=device)
    if not hasattr(model, "encode") or not hasattr(model, "state_head"):
        raise TypeError(f"{checkpoint_path} does not expose encode() and state_head.")

    stats = checkpoint.get("normalisation_stats")
    if stats is None:
        raise ValueError(f"{checkpoint_path} is missing normalisation_stats.")

    model.to(device)
    model.eval()
    dataset = SlidingWindowDataset(
        [str(test_csv)],
        model.get_window_size(),
        target_mode=model.get_target_type(),
        output_size=model.get_output_size(),
        output_offset=model.get_output_offset(),
        normalisation_stats=stats,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    probability_windows = []
    with torch.no_grad():
        for inputs, _targets in loader:
            inputs = inputs.to(device)
            logits = model.state_head(model.encode(inputs)).reshape(-1)
            probability_windows.append(torch.sigmoid(logits).detach().cpu().numpy())

    if not probability_windows:
        raise ValueError(f"No windows produced for {test_csv}.")

    raw_df = pd.read_csv(test_csv, low_memory=False)
    probabilities = np.concatenate(probability_windows, axis=0).astype(np.float32)
    reconstructed, coverage = reconstruct_series_from_windows(
        probabilities[:, None],
        len(raw_df),
        target_mode=model.get_target_type(),
        output_offset=model.get_output_offset(),
        output_size=model.get_output_size(),
        window_start_indices=dataset.get_window_locations(),
    )
    return reconstructed, coverage


def evaluate_one(appliance: str, checkpoint_path: Path, test_csv: Path, results_csv: Path, thresholds: list[float], batch_size: int) -> list[dict[str, float | str]]:
    rule = _get_status_rule(appliance)
    if rule is None:
        raise ValueError(f"No status rule found for {appliance}.")

    results = pd.read_csv(results_csv)
    state_prob, coverage = reconstruct_state_probabilities(checkpoint_path, test_csv, batch_size)
    valid_state_prob = state_prob[coverage > 0]
    if len(valid_state_prob) != len(results):
        raise ValueError(
            f"State probability length mismatch for {appliance}: "
            f"{len(valid_state_prob)} vs results {len(results)}"
        )

    prediction = results["prediction"].to_numpy(dtype=np.float32)
    ground_truth = results["ground truth"].to_numpy(dtype=np.float32)
    timestamps = results["time"].to_numpy()
    y_true = _power_to_status(ground_truth, rule, timestamps)

    min_threshold = float(rule["min_threshold"])
    max_threshold = float(rule.get("max_threshold", np.inf))
    power_on = (prediction > min_threshold) & (prediction <= max_threshold)

    rows: list[dict[str, float | str]] = []
    for threshold in thresholds:
        gated_on = power_on & (valid_state_prob >= threshold)
        metrics = binary_metrics(y_true, gated_on, valid_state_prob)
        rows.append(
            {
                "appliance": appliance,
                "model": "StateAwareSeq2Point+StateGate",
                "state_gate_threshold": float(threshold),
                "checkpoint": str(checkpoint_path),
                "test_csv": str(test_csv),
                "results_csv": str(results_csv),
                **metrics,
            }
        )
    return rows


def default_paths(appliance: str) -> tuple[Path, Path, Path]:
    checkpoint = (
        ROOT
        / "saved_models"
        / f"state_aware_official_{appliance}_exported_h1_h2_300k_strict_w0p05"
        / f"{appliance}_ukdale_exported_h1_h2_{appliance}_300k_strict_w0p05_StateAwareSeq2Point.pth"
    )
    test_csv = ROOT / "dataset" / "ukdale" / "exported_csv" / "UKDALE_dataset" / f"{appliance}_H2.csv"
    results_csv = (
        ROOT
        / "result"
        / "ukdale_exported_h1_h2"
        / f"{appliance}_300k_strict_w0p05"
        / f"{appliance}_StateAwareSeq2Point_results.csv"
    )
    return checkpoint, test_csv, results_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--appliances", nargs="+", default=list(DEFAULT_APPLIANCES))
    parser.add_argument("--thresholds", nargs="+", type=float, default=list(DEFAULT_THRESHOLDS))
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument(
        "--output",
        default=str(ROOT / "experiment" / "state_gate_multi_appliance_strict_w0p05_summary.csv"),
    )
    args = parser.parse_args()

    rows: list[dict[str, float | str]] = []
    for appliance in args.appliances:
        checkpoint, test_csv, results_csv = default_paths(appliance)
        print(f"Evaluating {appliance}")
        print(f"  checkpoint: {checkpoint}")
        rows.extend(evaluate_one(appliance, checkpoint, test_csv, results_csv, args.thresholds, args.batch_size))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved state-gate summary to {output}")


if __name__ == "__main__":
    main()
