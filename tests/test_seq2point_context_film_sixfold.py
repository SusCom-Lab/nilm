from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from experiments.run_seq2point_context_film import load_config
from experiments.run_seq2point_context_film_sixfold import (
    FOLD_ORDER,
    METRIC_COLUMNS,
    aggregate_sixfold_results,
    load_sixfold_config,
    validate_base_config,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SIXFOLD_CONFIG = (
    REPO_ROOT
    / "experiments/configs/refit_washing_machine_seq2point_film_sixfold_seed42.yaml"
)
BASE_CONFIG = (
    REPO_ROOT / "experiments/configs/refit_washing_machine_seq2point_film_dev.yaml"
)


class SixFoldExperimentTest(unittest.TestCase):
    def test_fixed_fold_definitions_and_hyperparameters(self):
        config = load_sixfold_config(SIXFOLD_CONFIG)
        validate_base_config(load_config(BASE_CONFIG))
        expected = {
            "F1": (["H3", "H5", "H6", "H7"], "H2", "H1"),
            "F2": (["H1", "H5", "H6", "H7"], "H3", "H2"),
            "F3": (["H1", "H2", "H5", "H6"], "H7", "H3"),
            "F4": (["H1", "H2", "H3", "H7"], "H6", "H5"),
            "F5": (["H1", "H2", "H3", "H7"], "H5", "H6"),
            "F6": (["H2", "H3", "H5", "H6"], "H1", "H7"),
        }
        for fold, (train, validation, test) in expected.items():
            self.assertEqual(config["folds"][fold]["train"], train)
            self.assertEqual(config["folds"][fold]["validation"], [validation])
            self.assertEqual(config["folds"][fold]["test"], [test])
            self.assertNotIn("H4", train + [validation, test])

    def test_aggregate_outputs_pairing_counts_wrong_details_and_leakage(self):
        config = load_sixfold_config(SIXFOLD_CONFIG)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for fold_index, fold in enumerate(FOLD_ORDER):
                split = config["folds"][fold]
                fold_dir = root / fold
                fold_dir.mkdir(parents=True)
                correct_mae = 9.0 if fold_index < 4 else 11.0
                wrong_mean_mae = 10.0 if fold_index != 3 else 8.0
                rows = [
                    self._result_row("M0 Seq2Point", 12.0, 0.60),
                    self._result_row("M1 Global FiLM", 10.0, 0.65),
                    self._result_row("M2 Correct Context", correct_mae, 0.70),
                ]
                for house in split["train"]:
                    rows.append(self._result_row(f"M3 Wrong {house}-A", wrong_mean_mae, 0.66))
                rows.extend(
                    [
                        self._result_row("M3 Wrong mean", wrong_mean_mae, 0.66),
                        self._result_row("M3 Wrong std", 0.2, 0.01),
                    ]
                )
                pd.DataFrame(rows).to_csv(fold_dir / "results.csv", index=False)
                timestamp_hash = f"hash-{fold}"
                manifest = {
                    "best_epochs": {"M0": 8, "M1": 2, "M2": 3, "M3": 3},
                    "checkpoints": {
                        "M2": f"{fold}/m2.pt",
                        "M3": f"{fold}/m2.pt",
                    },
                    "evaluation_timestamp_sha256": timestamp_hash,
                    "evaluation_timestamp_count": 100 + fold_index,
                    "training_history": {
                        mode: [{"epoch": 20}] for mode in ("M0", "M1", "M2")
                    },
                }
                (fold_dir / "manifest.json").write_text(
                    json.dumps(manifest), encoding="utf-8"
                )
                methods = [row["method"] for row in rows if row["method"] != "M3 Wrong std"]
                leakage = {
                    "passed": True,
                    "test_labels_read_after_all_predictions": True,
                    "test_not_in_normalisation": True,
                    "test_not_in_checkpoint_selection": True,
                    "wrong_query_house_block": f"{split['test'][0]}-B",
                    "wrong_query_object_reused": True,
                    "evaluation_timestamp_sha256": timestamp_hash,
                    "evaluation_timestamp_count": 100 + fold_index,
                    "all_methods_timestamp_sha256": {
                        method: timestamp_hash for method in methods
                    },
                }
                (fold_dir / "leakage_report.json").write_text(
                    json.dumps(leakage), encoding="utf-8"
                )

            output = aggregate_sixfold_results(root, config["folds"])
            self.assertEqual(
                output["win_counts"]["correct_beats_wrong_mean_households"], 3
            )
            self.assertEqual(
                output["win_counts"]["correct_beats_global_households"], 4
            )
            self.assertTrue(output["all_checks_passed"])
            self.assertEqual(len(pd.read_csv(root / "per_fold_results.csv")), 30)
            self.assertEqual(len(pd.read_csv(root / "wrong_context_details.csv")), 24)
            paired = pd.read_csv(root / "paired_household_results.csv")
            self.assertEqual(len(paired), 6)
            self.assertEqual(set(paired["test_house"]), {"H1", "H2", "H3", "H5", "H6", "H7"})
            with self.assertRaises(FileExistsError):
                aggregate_sixfold_results(root, config["folds"])

    @staticmethod
    def _result_row(method: str, mae: float, f1: float) -> dict:
        row = {"method": method}
        for metric in METRIC_COLUMNS:
            row[metric] = 1.0
        row["MAE"] = mae
        row["F1"] = f1
        return row


if __name__ == "__main__":
    unittest.main()
