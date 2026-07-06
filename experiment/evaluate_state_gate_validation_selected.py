"""
Select state-gate thresholds on H1 validation, then evaluate on H2.

This avoids choosing the gate threshold on the test set. For each appliance:

1. Rebuild the same H1 validation split used during training.
2. Evaluate state-gate thresholds on validation.
3. Select a threshold using a validation-only policy.
4. Apply the selected threshold once to H2 test results.
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


def select_split(df: pd.DataFrame, split_ratio: float, split_mode: str) -> pd.DataFrame:
    if "segment_id" in df.columns:
        segment_sizes = df.groupby("segment_id", sort=False).size().reset_index(name="rows")
        target_rows = len(df) * split_ratio
        cumulative_rows = segment_sizes["rows"].cumsum()
        split_index = int((cumulative_rows < target_rows).sum())
        if split_index < len(segment_sizes):
            current_gap = abs(cumulative_rows.iloc[split_index] - target_rows)
            previous_gap = abs((cumulative_rows.iloc[split_index - 1] if split_index > 0 else 0) - target_rows)
            if previous_gap < current_gap:
                split_index -= 1

        split_index = max(0, min(len(segment_sizes), split_index + 1))
        segment_ids = segment_sizes["segment_id"]
        if split_mode == "train":
            selected_segments = segment_ids.iloc[:split_index]
        elif split_mode == "val":
            selected_segments = segment_ids.iloc[split_index:]
        else:
            return df.copy()
        return df[df["segment_id"].isin(selected_segments)].copy()

    split_index = int(len(df) * split_ratio)
    if split_mode == "train":
        return df.iloc[:split_index].copy()
    if split_mode == "val":
        return df.iloc[split_index:].copy()
    return df.copy()


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


def default_paths(appliance: str) -> tuple[Path, Path, Path, Path]:
    checkpoint = (
        ROOT
        / "saved_models"
        / f"state_aware_official_{appliance}_exported_h1_h2_300k_strict_w0p05"
        / f"{appliance}_ukdale_exported_h1_h2_{appliance}_300k_strict_w0p05_StateAwareSeq2Point.pth"
    )
    train_csv = ROOT / "dataset" / "ukdale" / "exported_csv" / "UKDALE_dataset" / f"{appliance}_H1.csv"
    test_csv = ROOT / "dataset" / "ukdale" / "exported_csv" / "UKDALE_dataset" / f"{appliance}_H2.csv"
    test_results_csv = (
        ROOT
        / "result"
        / "ukdale_exported_h1_h2"
        / f"{appliance}_300k_strict_w0p05"
        / f"{appliance}_StateAwareSeq2Point_results.csv"
    )
    return checkpoint, train_csv, test_csv, test_results_csv


def load_state_model(checkpoint_path: Path, device: str):
    checkpoint = load_checkpoint(str(checkpoint_path), map_location=device)
    model = instantiate_from_checkpoint(checkpoint, map_location=device)
    if not hasattr(model, "encode") or not hasattr(model, "state_head"):
        raise TypeError(f"{checkpoint_path} does not expose encode() and state_head.")
    stats = checkpoint.get("normalisation_stats")
    if stats is None:
        raise ValueError(f"{checkpoint_path} is missing normalisation_stats.")
    model.to(device)
    model.eval()
    return checkpoint, model, stats


def reconstruct_test_state_probabilities(model, stats: dict, test_csv: Path, batch_size: int, device: str) -> tuple[np.ndarray, np.ndarray]:
    dataset = SlidingWindowDataset(
        [str(test_csv)],
        model.get_window_size(),
        target_mode=model.get_target_type(),
        output_size=model.get_output_size(),
        output_offset=model.get_output_offset(),
        normalisation_stats=stats,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    probabilities = []
    with torch.no_grad():
        for inputs, _targets in loader:
            inputs = inputs.to(device)
            logits = model.state_head(model.encode(inputs)).reshape(-1)
            probabilities.append(torch.sigmoid(logits).detach().cpu().numpy())

    raw_df = pd.read_csv(test_csv, low_memory=False)
    probability_windows = np.concatenate(probabilities, axis=0).astype(np.float32)
    return reconstruct_series_from_windows(
        probability_windows[:, None],
        len(raw_df),
        target_mode=model.get_target_type(),
        output_offset=model.get_output_offset(),
        output_size=model.get_output_size(),
        window_start_indices=dataset.get_window_locations(),
    )


def evaluate_validation(appliance: str, model, stats: dict, train_csv: Path, crop: int, val_ratio: float, batch_size: int, device: str):
    split_ratio = 1.0 - val_ratio
    full_df = pd.read_csv(train_csv, low_memory=False)
    if crop:
        full_df = full_df.head(crop)
    val_df = select_split(full_df, split_ratio, "val").reset_index(drop=True)

    dataset = SlidingWindowDataset(
        [str(train_csv)],
        model.get_window_size(),
        crop=crop,
        split_ratio=split_ratio,
        split_mode="val",
        target_mode=model.get_target_type(),
        output_size=model.get_output_size(),
        output_offset=model.get_output_offset(),
        normalisation_stats=stats,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    power_windows = []
    probability_windows = []
    with torch.no_grad():
        for inputs, _targets in loader:
            inputs = inputs.to(device)
            z = model.encode(inputs)
            power = model.prepare_outputs(model.power_head(z)).detach().cpu().numpy()
            logits = model.state_head(z).reshape(-1)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            power_windows.append(power)
            probability_windows.append(probs)

    if not power_windows:
        raise ValueError(f"No validation windows produced for {appliance}.")

    pred_norm = np.concatenate(power_windows, axis=0).astype(np.float32)
    probs = np.concatenate(probability_windows, axis=0).astype(np.float32)
    pred_reconstructed, coverage = reconstruct_series_from_windows(
        pred_norm[:, None],
        len(val_df),
        target_mode=model.get_target_type(),
        output_offset=model.get_output_offset(),
        output_size=model.get_output_size(),
        window_start_indices=dataset.get_window_locations(),
    )
    prob_reconstructed, prob_coverage = reconstruct_series_from_windows(
        probs[:, None],
        len(val_df),
        target_mode=model.get_target_type(),
        output_offset=model.get_output_offset(),
        output_size=model.get_output_size(),
        window_start_indices=dataset.get_window_locations(),
    )
    valid_mask = (coverage > 0) & (prob_coverage > 0)
    prediction = (pred_reconstructed[valid_mask] * float(stats["appliance_std"])) + float(stats["appliance_mean"])
    prediction = np.clip(prediction, 0.0, None)
    state_prob = prob_reconstructed[valid_mask]
    timestamps = val_df.iloc[:, 0].to_numpy()[valid_mask]
    ground_truth = val_df.iloc[:, 2].astype(float).to_numpy()[valid_mask]
    return prediction, state_prob, ground_truth, timestamps


def evaluate_test(model, stats: dict, test_csv: Path, test_results_csv: Path, batch_size: int, device: str):
    state_prob, coverage = reconstruct_test_state_probabilities(model, stats, test_csv, batch_size, device)
    results = pd.read_csv(test_results_csv)
    valid_state_prob = state_prob[coverage > 0]
    if len(valid_state_prob) != len(results):
        raise ValueError(f"State probability length mismatch: {len(valid_state_prob)} vs {len(results)}")
    prediction = results["prediction"].to_numpy(dtype=np.float32)
    ground_truth = results["ground truth"].to_numpy(dtype=np.float32)
    timestamps = results["time"].to_numpy()
    return prediction, valid_state_prob, ground_truth, timestamps


def threshold_rows(appliance: str, split: str, prediction, state_prob, ground_truth, timestamps, thresholds: list[float]) -> list[dict[str, float | str]]:
    rule = _get_status_rule(appliance)
    if rule is None:
        raise ValueError(f"No status rule found for {appliance}.")

    y_true = _power_to_status(ground_truth, rule, timestamps)
    min_threshold = float(rule["min_threshold"])
    max_threshold = float(rule.get("max_threshold", np.inf))
    power_on = (prediction > min_threshold) & (prediction <= max_threshold)
    power_metrics = binary_metrics(y_true, power_on, prediction)

    rows = []
    for threshold in thresholds:
        gated_on = power_on & (state_prob >= threshold)
        metrics = binary_metrics(y_true, gated_on, state_prob)
        rows.append(
            {
                "appliance": appliance,
                "split": split,
                "state_gate_threshold": float(threshold),
                "power_only_Recall": power_metrics["Recall"],
                "power_only_FPR": power_metrics["FPR"],
                "power_only_F1-score": power_metrics["F1-score"],
                **metrics,
            }
        )
    return rows


def choose_threshold(rows: list[dict[str, float | str]], policy: str, max_recall_drop: float) -> tuple[float, str]:
    frame = pd.DataFrame(rows)
    if policy == "max_f1":
        chosen = frame.sort_values(["F1-score", "FPR"], ascending=[False, True]).iloc[0]
        return float(chosen["state_gate_threshold"]), "max_f1"

    baseline_recall = float(frame["power_only_Recall"].iloc[0])
    min_recall = baseline_recall * (1.0 - max_recall_drop)
    eligible = frame[frame["Recall"] >= min_recall]
    if eligible.empty:
        chosen = frame.sort_values(["F1-score", "FPR"], ascending=[False, True]).iloc[0]
        return float(chosen["state_gate_threshold"]), "fallback_max_f1"

    chosen = eligible.sort_values(["FPR", "F1-score"], ascending=[True, False]).iloc[0]
    return float(chosen["state_gate_threshold"]), "min_fpr_under_recall_drop"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--appliances", nargs="+", default=list(DEFAULT_APPLIANCES))
    parser.add_argument("--thresholds", nargs="+", type=float, default=list(DEFAULT_THRESHOLDS))
    parser.add_argument("--crop", type=int, default=300000)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument(
        "--selection-policy",
        choices=("min_fpr_under_recall_drop", "max_f1"),
        default="min_fpr_under_recall_drop",
    )
    parser.add_argument("--max-recall-drop", type=float, default=0.10)
    parser.add_argument(
        "--validation-output",
        default=str(ROOT / "experiment" / "state_gate_validation_threshold_grid_strict_w0p05.csv"),
    )
    parser.add_argument(
        "--test-output",
        default=str(ROOT / "experiment" / "state_gate_validation_selected_h2_strict_w0p05_summary.csv"),
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    validation_rows: list[dict[str, float | str]] = []
    test_rows: list[dict[str, float | str]] = []

    for appliance in args.appliances:
        checkpoint_path, train_csv, test_csv, test_results_csv = default_paths(appliance)
        print(f"Evaluating {appliance}")
        _checkpoint, model, stats = load_state_model(checkpoint_path, device)

        val_prediction, val_state_prob, val_ground_truth, val_timestamps = evaluate_validation(
            appliance, model, stats, train_csv, args.crop, args.val_ratio, args.batch_size, device
        )
        app_val_rows = threshold_rows(
            appliance,
            "validation",
            val_prediction,
            val_state_prob,
            val_ground_truth,
            val_timestamps,
            args.thresholds,
        )
        selected_threshold, selection_reason = choose_threshold(
            app_val_rows, args.selection_policy, args.max_recall_drop
        )
        for row in app_val_rows:
            row["selected"] = float(row["state_gate_threshold"]) == selected_threshold
            row["selection_reason"] = selection_reason
            validation_rows.append(row)

        test_prediction, test_state_prob, test_ground_truth, test_timestamps = evaluate_test(
            model, stats, test_csv, test_results_csv, args.batch_size, device
        )
        selected_test_row = [
            row
            for row in threshold_rows(
                appliance,
                "test",
                test_prediction,
                test_state_prob,
                test_ground_truth,
                test_timestamps,
                [selected_threshold],
            )
        ][0]
        selected_test_row["selection_reason"] = selection_reason
        test_rows.append(selected_test_row)
        print(
            f"  selected threshold={selected_threshold:.3f} ({selection_reason}), "
            f"test FPR={selected_test_row['FPR']:.6f}, "
            f"test Recall={selected_test_row['Recall']:.6f}, "
            f"test F1={selected_test_row['F1-score']:.6f}"
        )

    validation_output = Path(args.validation_output)
    validation_output.parent.mkdir(parents=True, exist_ok=True)
    with validation_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(validation_rows[0].keys()))
        writer.writeheader()
        writer.writerows(validation_rows)

    test_output = Path(args.test_output)
    test_output.parent.mkdir(parents=True, exist_ok=True)
    with test_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(test_rows[0].keys()))
        writer.writeheader()
        writer.writerows(test_rows)

    print(f"Saved validation threshold grid to {validation_output}")
    print(f"Saved selected-threshold H2 summary to {test_output}")


if __name__ == "__main__":
    main()
