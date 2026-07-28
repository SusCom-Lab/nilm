#!/usr/bin/env python3
"""Run and aggregate the fixed seed-42 household-level six-fold experiment."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.run_seq2point_context_film import (  # noqa: E402
    Experiment,
    _git_state,
    _json_dump,
    load_config,
)


FOLD_ORDER = ("F1", "F2", "F3", "F4", "F5", "F6")
HOUSE_SET = {"H1", "H2", "H3", "H5", "H6", "H7"}
SUMMARY_METHODS = (
    "M0 Seq2Point",
    "M1 Global FiLM",
    "M2 Correct Context",
    "M3 Wrong mean",
    "M3 Wrong std",
)
AGGREGATE_METHODS = SUMMARY_METHODS[:-1]
METRIC_COLUMNS = (
    "MAE",
    "MAE-on",
    "MAE-off",
    "SAE",
    "Precision",
    "Recall",
    "F1",
    "FPR",
    "true_total_energy_Wh",
    "pred_total_energy_Wh",
    "true_ON_mean_power_W",
    "pred_ON_mean_power_W",
    "gamma_norm",
    "beta_norm",
)
EXPECTED_BASE_VALUES = {
    "seed": 42,
    "window_size": 599,
    "hidden_dim": 1024,
    "context_k": 16,
    "context_selector": "uniform",
    "context_code_dim": 128,
    "generator_hidden_dim": 256,
}
EXPECTED_TRAINING_VALUES = {
    "batch_size": 256,
    "baseline_epochs": 30,
    "global_film_epochs": 20,
    "context_film_epochs": 20,
    "minimum_epochs": 20,
    "learning_rate": 0.001,
    "adapter_learning_rate": 0.001,
    "patience": 8,
    "min_delta": 0.0001,
    "num_workers": 0,
}


def load_sixfold_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if tuple(config["folds"]) != FOLD_ORDER:
        raise ValueError(f"Fold order must be exactly {FOLD_ORDER}.")
    if int(config["seed"]) != 42:
        raise ValueError("This first six-fold experiment is fixed to seed=42.")
    if set(config.get("excluded_houses", [])) != {"H4"}:
        raise ValueError("H4 must be the only explicitly excluded household.")
    test_houses = []
    for fold_name in FOLD_ORDER:
        split = config["folds"][fold_name]
        train = list(split["train"])
        validation = list(split["validation"])
        test = list(split["test"])
        if len(train) != 4 or len(validation) != 1 or len(test) != 1:
            raise ValueError(f"{fold_name} must contain 4 train, 1 validation, and 1 test house.")
        groups = [set(train), set(validation), set(test)]
        if any(left.intersection(right) for i, left in enumerate(groups) for right in groups[i + 1 :]):
            raise ValueError(f"{fold_name} household roles overlap.")
        if set(train + validation + test) != HOUSE_SET:
            raise ValueError(f"{fold_name} must cover exactly {sorted(HOUSE_SET)}.")
        test_houses.extend(test)
    if set(test_houses) != HOUSE_SET or len(test_houses) != len(HOUSE_SET):
        raise ValueError("Each eligible household must be the test household exactly once.")
    return config


def validate_base_config(base: dict) -> None:
    for key, expected in EXPECTED_BASE_VALUES.items():
        if base.get(key) != expected:
            raise ValueError(f"Base config {key}={base.get(key)!r}; expected {expected!r}.")
    for key, expected in EXPECTED_TRAINING_VALUES.items():
        if base["training"].get(key) != expected:
            raise ValueError(
                f"Base training config {key}={base['training'].get(key)!r}; expected {expected!r}."
            )
    if base.get("appliance") != "washing_machine":
        raise ValueError("Six-fold experiment is fixed to washing_machine.")
    if set(base["fixed_14_day_starts"]) != HOUSE_SET:
        raise ValueError("Base manifest starts must contain exactly H1,H2,H3,H5,H6,H7.")


def materialize_fold_config(
    sixfold_config: dict,
    base_config: dict,
    fold_name: str,
) -> Path:
    artifact_root = REPO_ROOT / sixfold_config["artifacts_root"]
    config_path = artifact_root / "fold_configs" / f"{fold_name}.yaml"
    fold_config = deepcopy(base_config)
    fold_config["experiment_name"] = (
        f"{sixfold_config['experiment_name']}_{fold_name.lower()}"
    )
    fold_config["house_split"] = deepcopy(sixfold_config["folds"][fold_name])
    fold_config["artifacts_root"] = sixfold_config["artifacts_root"]
    fold_config["default_run_id"] = fold_name
    fold_config["sixfold"] = {
        "fold": fold_name,
        "seed": 42,
        "parent_config": str(
            Path("experiments/configs")
            / "refit_washing_machine_seq2point_film_sixfold_seed42.yaml"
        ),
    }
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing != fold_config:
            raise RuntimeError(
                f"Existing materialized config differs: {config_path}. "
                "Use a new artifacts_root instead of overwriting it."
            )
    else:
        _json_dump(config_path, fold_config)
    return config_path


def _load_manifest(experiment: Experiment) -> dict | None:
    if not experiment.manifest_path.exists():
        return None
    with experiment.manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _checkpoint_stage_complete(
    experiment: Experiment,
    mode: str,
    checkpoint_path: Path,
) -> bool:
    manifest = _load_manifest(experiment)
    if not checkpoint_path.exists():
        if manifest is not None and mode in manifest.get("checkpoints", {}):
            raise RuntimeError(f"{mode} manifest entry exists but checkpoint is missing.")
        return False
    if manifest is None or mode not in manifest.get("checkpoints", {}):
        raise RuntimeError(
            f"Incomplete {mode} stage in {experiment.run_dir}: checkpoint exists "
            "without a completed manifest entry. Do not silently reuse it."
        )
    recorded = REPO_ROOT / manifest["checkpoints"][mode]
    if recorded.resolve() != checkpoint_path.resolve():
        raise RuntimeError(f"{mode} manifest checkpoint does not match {checkpoint_path}.")
    return True


def run_fold(experiment: Experiment) -> None:
    fold_name = experiment.run_id
    print(f"\n===== {fold_name}: {experiment.config['house_split']} =====")
    if not _checkpoint_stage_complete(experiment, "M0", experiment.baseline_path):
        experiment.train_baseline()
    else:
        print(f"{fold_name} M0 already complete; skipping.")
    if not _checkpoint_stage_complete(experiment, "M1", experiment.global_path):
        experiment.train_global_film()
    else:
        print(f"{fold_name} M1 already complete; skipping.")
    if not _checkpoint_stage_complete(experiment, "M2", experiment.context_path):
        experiment.train_context_film()
    else:
        print(f"{fold_name} M2/M3 checkpoint already complete; skipping.")

    required_evaluation = (
        experiment.run_dir / "results.csv",
        experiment.run_dir / "summary.json",
        experiment.run_dir / "leakage_report.json",
    )
    existing = [path.exists() for path in required_evaluation]
    if all(existing):
        print(f"{fold_name} evaluation already complete; skipping.")
    elif any(existing):
        raise RuntimeError(
            f"Incomplete evaluation outputs in {experiment.run_dir}; "
            "existing results are not overwritten."
        )
    else:
        experiment.evaluate_all()


def _method_row(results: pd.DataFrame, method: str) -> pd.Series:
    selected = results[results["method"] == method]
    if len(selected) != 1:
        raise RuntimeError(f"Expected exactly one {method!r} result, found {len(selected)}.")
    return selected.iloc[0]


def aggregate_sixfold_results(
    artifact_root: Path,
    folds: dict,
    *,
    refuse_existing: bool = True,
) -> dict:
    def recorded_path(path: Path) -> str:
        try:
            return str(path.relative_to(REPO_ROOT))
        except ValueError:
            return str(path)

    output_paths = {
        "per_fold_results": artifact_root / "per_fold_results.csv",
        "wrong_context_details": artifact_root / "wrong_context_details.csv",
        "summary_mean_std": artifact_root / "summary_mean_std.csv",
        "paired_household_results": artifact_root / "paired_household_results.csv",
        "win_counts": artifact_root / "win_counts.json",
        "leakage_checks": artifact_root / "leakage_timestamp_checks.json",
        "manifest": artifact_root / "sixfold_manifest.json",
    }
    if refuse_existing:
        existing = [str(path) for path in output_paths.values() if path.exists()]
        if existing:
            raise FileExistsError(
                "Refusing to overwrite existing six-fold summaries: " + ", ".join(existing)
            )

    per_fold_rows = []
    wrong_rows = []
    paired_rows = []
    leakage_rows = []
    fold_manifests = {}
    for fold_name in FOLD_ORDER:
        fold_dir = artifact_root / fold_name
        required = (
            fold_dir / "results.csv",
            fold_dir / "manifest.json",
            fold_dir / "leakage_report.json",
        )
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(f"{fold_name} is incomplete: {missing}")
        results = pd.read_csv(required[0])
        manifest = json.loads(required[1].read_text(encoding="utf-8"))
        leakage = json.loads(required[2].read_text(encoding="utf-8"))
        fold_manifests[fold_name] = {
            "manifest": recorded_path(required[1]),
            "best_epochs": manifest["best_epochs"],
            "evaluation_timestamp_sha256": manifest["evaluation_timestamp_sha256"],
            "evaluation_timestamp_count": manifest["evaluation_timestamp_count"],
        }
        split = folds[fold_name]
        test_house = split["test"][0]
        common = {
            "fold": fold_name,
            "train_houses": ",".join(split["train"]),
            "validation_house": split["validation"][0],
            "test_house": test_house,
        }
        for method in SUMMARY_METHODS:
            row = _method_row(results, method)
            per_fold_rows.append(
                {**common, "method": method, **{metric: row[metric] for metric in METRIC_COLUMNS}}
            )

        wrong_details = results[
            results["method"].str.startswith("M3 Wrong H")
        ].copy()
        if len(wrong_details) != 4:
            raise RuntimeError(f"{fold_name} must contain four individual Wrong results.")
        for _, row in wrong_details.iterrows():
            context_house = row["method"].removeprefix("M3 Wrong ").removesuffix("-A")
            wrong_rows.append(
                {
                    **common,
                    "wrong_context_house": context_house,
                    "query_house_block": f"{test_house}-B",
                    **{metric: row[metric] for metric in METRIC_COLUMNS},
                }
            )
            if context_house not in split["train"]:
                raise RuntimeError(f"{fold_name} Wrong context {context_house} is not a train house.")

        global_row = _method_row(results, "M1 Global FiLM")
        correct_row = _method_row(results, "M2 Correct Context")
        wrong_mean = _method_row(results, "M3 Wrong mean")
        paired_rows.append(
            {
                **common,
                "global_mae": global_row["MAE"],
                "correct_mae": correct_row["MAE"],
                "wrong_mean_mae": wrong_mean["MAE"],
                "delta_mae_vs_wrong": correct_row["MAE"] - wrong_mean["MAE"],
                "delta_mae_vs_global": correct_row["MAE"] - global_row["MAE"],
                "global_f1": global_row["F1"],
                "correct_f1": correct_row["F1"],
                "wrong_mean_f1": wrong_mean["F1"],
                "delta_f1_vs_wrong": correct_row["F1"] - wrong_mean["F1"],
            }
        )

        timestamp_hashes = leakage["all_methods_timestamp_sha256"]
        timestamp_match = len(set(timestamp_hashes.values())) == 1
        history = manifest["training_history"]
        trained_epochs = {
            mode: max(int(row["epoch"]) for row in rows)
            for mode, rows in history.items()
        }
        leakage_rows.append(
            {
                "fold": fold_name,
                "test_house": test_house,
                "passed": bool(leakage["passed"]),
                "labels_after_predictions": bool(
                    leakage["test_labels_read_after_all_predictions"]
                ),
                "test_not_in_normalisation": bool(leakage["test_not_in_normalisation"]),
                "test_not_in_checkpoint_selection": bool(
                    leakage["test_not_in_checkpoint_selection"]
                ),
                "wrong_query_fixed": leakage["wrong_query_house_block"] == f"{test_house}-B",
                "wrong_query_object_reused": bool(leakage["wrong_query_object_reused"]),
                "shared_m2_m3_checkpoint": (
                    manifest["checkpoints"]["M2"] == manifest["checkpoints"]["M3"]
                ),
                "common_timestamps": timestamp_match,
                "timestamp_sha256": leakage["evaluation_timestamp_sha256"],
                "timestamp_count": int(leakage["evaluation_timestamp_count"]),
                "m0_trained_epochs": trained_epochs["M0"],
                "m1_trained_epochs": trained_epochs["M1"],
                "m2_trained_epochs": trained_epochs["M2"],
                "minimum_epochs_satisfied": all(
                    trained_epochs[mode] >= 20 for mode in ("M0", "M1", "M2")
                ),
            }
        )

    per_fold = pd.DataFrame(per_fold_rows)
    wrong_details = pd.DataFrame(wrong_rows)
    paired = pd.DataFrame(paired_rows).sort_values("test_house").reset_index(drop=True)
    summary_rows = []
    for method in AGGREGATE_METHODS:
        selected = per_fold[per_fold["method"] == method]
        summary = {"method": method, "folds": len(selected)}
        for metric in METRIC_COLUMNS:
            values = selected[metric].to_numpy(dtype=np.float64)
            summary[f"{metric}_mean"] = float(np.mean(values))
            summary[f"{metric}_std"] = float(np.std(values, ddof=0))
        summary_rows.append(summary)
    summary_mean_std = pd.DataFrame(summary_rows)
    leakage_frame = pd.DataFrame(leakage_rows)
    correct_beats_wrong = int((paired["delta_mae_vs_wrong"] < 0).sum())
    correct_beats_global = int((paired["delta_mae_vs_global"] < 0).sum())
    win_counts = {
        "comparison_metric": "MAE (lower is better)",
        "correct_beats_wrong_mean_households": correct_beats_wrong,
        "correct_beats_global_households": correct_beats_global,
        "total_test_households": 6,
    }
    all_leakage_passed = bool(
        leakage_frame[
            [
                "passed",
                "labels_after_predictions",
                "test_not_in_normalisation",
                "test_not_in_checkpoint_selection",
                "wrong_query_fixed",
                "wrong_query_object_reused",
                "shared_m2_m3_checkpoint",
                "common_timestamps",
                "minimum_epochs_satisfied",
            ]
        ].to_numpy(dtype=bool).all()
    )
    leakage_output = {
        "all_folds_passed": all_leakage_passed,
        "folds": leakage_frame.to_dict(orient="records"),
    }
    sixfold_manifest = {
        "experiment": "REFIT washing_machine Seq2Point Context-FiLM household six-fold",
        "seed": 42,
        "fold_order": list(FOLD_ORDER),
        "folds": folds,
        "git": _git_state(),
        "fold_manifests": fold_manifests,
        "outputs": {name: recorded_path(path) for name, path in output_paths.items()},
        "win_counts": win_counts,
        "all_leakage_and_timestamp_checks_passed": all_leakage_passed,
        "hyperparameters_modified_after_run": False,
    }
    artifact_root.mkdir(parents=True, exist_ok=True)
    per_fold.to_csv(output_paths["per_fold_results"], index=False)
    wrong_details.to_csv(output_paths["wrong_context_details"], index=False)
    summary_mean_std.to_csv(output_paths["summary_mean_std"], index=False)
    paired.to_csv(output_paths["paired_household_results"], index=False)
    _json_dump(output_paths["win_counts"], win_counts)
    _json_dump(output_paths["leakage_checks"], leakage_output)
    _json_dump(output_paths["manifest"], sixfold_manifest)
    return {
        "paths": {name: str(path) for name, path in output_paths.items()},
        "win_counts": win_counts,
        "all_checks_passed": all_leakage_passed,
        "paired": paired,
    }


class SixFoldRunner:
    def __init__(self, config_path: Path, device: str | None):
        self.config_path = config_path.resolve()
        self.config = load_sixfold_config(self.config_path)
        base_path = REPO_ROOT / self.config["base_config"]
        self.base_config = load_config(base_path)
        validate_base_config(self.base_config)
        self.device = device
        self.artifact_root = REPO_ROOT / self.config["artifacts_root"]

    def experiment(self, fold_name: str) -> Experiment:
        config_path = materialize_fold_config(
            self.config, self.base_config, fold_name
        )
        return Experiment(config_path, fold_name, self.device)

    def run_fold(self, fold_name: str) -> None:
        run_fold(self.experiment(fold_name))

    def aggregate(self) -> dict:
        return aggregate_sixfold_results(
            self.artifact_root, self.config["folds"], refuse_existing=True
        )

    def run_all(self) -> dict:
        final_summary = self.artifact_root / "sixfold_manifest.json"
        if final_summary.exists():
            raise FileExistsError(
                f"Six-fold aggregation already exists: {final_summary}. "
                "Existing results are never overwritten."
            )
        for fold_name in FOLD_ORDER:
            self.run_fold(fold_name)
        return self.aggregate()


def build_parser() -> argparse.ArgumentParser:
    default_config = (
        REPO_ROOT
        / "experiments/configs/refit_washing_machine_seq2point_film_sixfold_seed42.yaml"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run_all", "run_fold", "aggregate"))
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--fold", choices=FOLD_ORDER)
    parser.add_argument("--device", help="For example cuda:3 (default: auto).")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    runner = SixFoldRunner(args.config, args.device)
    if args.command == "run_fold":
        if args.fold is None:
            raise SystemExit("--fold is required for run_fold.")
        runner.run_fold(args.fold)
        return
    if args.fold is not None:
        raise SystemExit("--fold is only valid with run_fold.")
    if args.command == "aggregate":
        output = runner.aggregate()
    else:
        output = runner.run_all()
    print(output["paired"].to_string(index=False))
    print(json.dumps(output["win_counts"], indent=2))
    print(f"all_checks_passed={output['all_checks_passed']}")


if __name__ == "__main__":
    main()
