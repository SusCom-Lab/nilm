"""
Official-pipeline comparison for Seq2Point and StateAwareSeq2Point.

This script uses the project's Trainer and Evaluator so results follow the
normal checkpointing, sequence reconstruction, duration-rule status metrics,
and result/metrics CSV output paths.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model_pipeline.models.seq2point.seq2point_state_aware import StateAwareSeq2Point, power_to_status
from model_pipeline.models.seq2point.seq2point import Seq2Point
from model_pipeline.test_model import Evaluator
from model_pipeline.train_model import Trainer, build_windowed_loaders, set_random_seed


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
    return float(max(0, total - positives) / max(1, positives))


def train_and_evaluate_baseline(args) -> None:
    set_random_seed(args.seed)
    train_loader, validation_loader, stats = build_windowed_loaders(
        [args.train_csv],
        window_size=args.window_size,
        target_mode="point",
        output_size=1,
        output_offset=args.window_size // 2,
        crop=args.crop,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    model = Seq2Point(window_size=args.window_size, hidden_dim=args.hidden_dim)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        appliance=args.appliance,
        dataset=args.dataset,
        model_save_dir=args.model_save_dir,
        result_dir=args.result_dir,
        seed=args.seed,
        batch_size=args.batch_size,
        crop=args.crop,
        normalisation_stats=stats,
    )
    trainer.trainModel(args.epochs)
    checkpoint = Path(args.model_save_dir) / f"{args.appliance}_{args.dataset}_Seq2Point.pth"
    evaluate_checkpoint(checkpoint, args)


def train_and_evaluate_state_aware(args) -> None:
    set_random_seed(args.seed)
    raw_threshold = load_status_threshold(args.appliance)
    train_loader, validation_loader, stats = build_windowed_loaders(
        [args.train_csv],
        window_size=args.window_size,
        target_mode="point",
        output_size=1,
        output_offset=args.window_size // 2,
        crop=args.crop,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    state_threshold = normalise_threshold(raw_threshold, stats)
    state_pos_weight = estimate_pos_weight(train_loader, state_threshold)
    print(
        f"State-aware settings: raw_threshold={raw_threshold:.3f}, "
        f"normalised_threshold={state_threshold:.6f}, pos_weight={state_pos_weight:.3f}"
    )

    model = StateAwareSeq2Point(
        window_size=args.window_size,
        hidden_dim=args.hidden_dim,
        state_threshold=state_threshold,
        state_loss_weight=args.state_loss_weight,
        state_pos_weight=state_pos_weight,
    )
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        appliance=args.appliance,
        dataset=args.dataset,
        model_save_dir=args.model_save_dir,
        result_dir=args.result_dir,
        seed=args.seed,
        batch_size=args.batch_size,
        crop=args.crop,
        normalisation_stats=stats,
    )
    trainer.trainModel(args.epochs)
    checkpoint = Path(args.model_save_dir) / f"{args.appliance}_{args.dataset}_StateAwareSeq2Point.pth"
    evaluate_checkpoint(checkpoint, args)


def evaluate_checkpoint(checkpoint: Path, args) -> None:
    evaluator = Evaluator(
        model_state_dir=str(checkpoint),
        test_csv_dir=args.test_csv,
        dataset=args.dataset,
        result_dir=args.result_dir,
        batch_size=args.batch_size,
    )
    evaluator.testModel()
    evaluator.saveMetrics()
    results_path = Path(args.result_dir) / f"{evaluator.appliance_name_formatted}_{evaluator.model_name}_results.csv"
    evaluator.getResults().to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")
    print(f"Evaluated {checkpoint}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--appliance", default="kettle")
    parser.add_argument("--dataset", default="ukdale_h1_h2")
    parser.add_argument("--train-csv", default=str(ROOT / "dataset" / "UKDALE_dataset" / "kettle_H1.csv"))
    parser.add_argument("--test-csv", default=str(ROOT / "dataset" / "UKDALE_dataset" / "kettle_H2.csv"))
    parser.add_argument("--crop", type=int, default=100000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--window-size", type=int, default=599)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--state-loss-weight", type=float, default=0.1)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-save-dir", default=str(ROOT / "saved_models" / "state_aware_official"))
    parser.add_argument("--result-dir", default=str(ROOT / "result" / "ukdale_h1_h2" / "kettle_state_aware_official"))
    parser.add_argument(
        "--models",
        choices=("seq2point", "state_aware_seq2point", "both"),
        default="both",
    )
    args = parser.parse_args()

    Path(args.model_save_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
    set_random_seed(args.seed)

    if args.models in ("seq2point", "both"):
        train_and_evaluate_baseline(args)
    if args.models in ("state_aware_seq2point", "both"):
        train_and_evaluate_state_aware(args)


if __name__ == "__main__":
    main()
