"""Public full-timeline evaluator shared by every NILM model plugin."""

from __future__ import annotations

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from model_pipeline.api import InferenceContext
from model_pipeline.data_protocol import (
    build_evaluation_target,
    get_appliance_status_rule,
    hide_partition_targets,
    load_supervised_partition,
)
from model_pipeline.model_registry import create_model, load_checkpoint
from model_pipeline.train_model import _aligned_prediction


METRIC_NAMES = ("MAE", "SAE", "Precision", "Recall", "F1", "MAE-on", "MAE-off")


def _status_threshold(appliance: str) -> float:
    rule = get_appliance_status_rule(appliance)
    if rule is None:
        raise KeyError(
            f"No public status threshold is configured for appliance '{appliance}'."
        )
    return rule["status_threshold"]


def compute_metrics(
    prediction: np.ndarray,
    truth: np.ndarray,
    status: np.ndarray,
    *,
    threshold: float,
) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=np.float32)
    truth = np.asarray(truth, dtype=np.float32)
    true_status = np.asarray(status, dtype=bool)
    predicted_status = prediction >= threshold
    absolute_error = np.abs(prediction - truth)
    true_positive = int(np.sum(predicted_status & true_status))
    false_positive = int(np.sum(predicted_status & ~true_status))
    false_negative = int(np.sum(~predicted_status & true_status))
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    return {
        "MAE": float(np.mean(absolute_error)),
        "SAE": float(
            abs(float(np.sum(prediction)) - float(np.sum(truth)))
            / max(float(np.sum(np.abs(truth))), 1e-8)
        ),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(2 * precision * recall / max(precision + recall, 1e-8)),
        "MAE-on": float(np.mean(absolute_error[true_status])) if true_status.any() else np.nan,
        "MAE-off": float(np.mean(absolute_error[~true_status])) if (~true_status).any() else np.nan,
    }


class Evaluator:
    def __init__(
        self,
        *,
        model_state_dir: str,
        test_csv_dir: str | list[str],
        dataset: str | None = None,
        result_dir: str | None = None,
        batch_size: int | None = None,
        device: str | None = None,
        **_legacy,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint = load_checkpoint(model_state_dir, map_location=self.device)
        self.model = create_model(
            self.checkpoint["model_key"], **self.checkpoint.get("init_kwargs", {})
        )
        self.model.load(model_state_dir, self.device)
        self.model_name = getattr(self.model, "display_name", self.checkpoint["model_key"])
        self.metadata = dict(self.checkpoint.get("metadata", {}))
        self.dataset = (dataset or self.metadata.get("dataset", "unknown")).lower()
        self.appliance_name_formatted = str(
            self.metadata.get("appliance", "multi_appliance")
        ).replace(" ", "_")
        self.result_dir = result_dir or os.path.join(
            "result", self.dataset, self.appliance_name_formatted
        )
        os.makedirs(self.result_dir, exist_ok=True)
        paths = [test_csv_dir] if isinstance(test_csv_dir, str) else list(test_csv_dir)
        self.test_data = load_supervised_partition("test", paths)
        forbidden = set(self.test_data.household_ids).intersection(
            self.metadata.get("train_households", []) + self.metadata.get("validation_households", [])
        )
        if forbidden:
            raise ValueError(f"Test household leakage detected: {sorted(forbidden)}.")
        self.test_input = hide_partition_targets(self.test_data)
        self.target = build_evaluation_target(self.test_data)
        self.aggregate = np.concatenate([series.aggregate for series in self.test_data.series])
        self.batch_size = int(
            batch_size or getattr(self.model, "official_batch_size", 512)
        )
        self.prediction = None
        self.metrics = None
        self.inference_time = None

    def testModel(self):
        started = time.perf_counter()
        raw_prediction = self.model.predict(
            self.test_input,
            InferenceContext(device=self.device, batch_size=self.batch_size),
        )
        self.inference_time = time.perf_counter() - started
        self.prediction = _aligned_prediction(raw_prediction, self.target, self.aggregate)
        rows = []
        for index, appliance in enumerate(self.target.appliances):
            threshold = _status_threshold(appliance)
            status = (
                self.target.status[:, index]
                if self.target.status is not None
                else self.target.appliance_power[:, index] >= threshold
            )
            rows.append(
                {
                    "appliance": appliance,
                    "model": self.model_name,
                    "dataset": self.dataset,
                    **compute_metrics(
                        self.prediction[:, index],
                        np.clip(self.target.appliance_power[:, index], 0.0, None),
                        status,
                        threshold=threshold,
                    ),
                }
            )
        self.metrics = pd.DataFrame(rows)
        return self.prediction

    def evaluate(self):
        self.testModel()
        return self.getMetrics()

    def getResults(self) -> pd.DataFrame:
        if self.prediction is None:
            self.testModel()
        data = {
            "time": self.target.timestamps,
            "household": self.target.household_ids,
            "aggregate": self.aggregate,
        }
        for index, appliance in enumerate(self.target.appliances):
            data[f"{appliance}_prediction"] = self.prediction[:, index]
            data[f"{appliance}_ground_truth"] = self.target.appliance_power[:, index]
        return pd.DataFrame(data)

    def getMetrics(self) -> pd.DataFrame:
        if self.metrics is None:
            self.testModel()
        return self.metrics.copy()

    def saveMetrics(self) -> pd.DataFrame:
        metrics = self.getMetrics()
        path = os.path.join(
            self.result_dir, f"{self.appliance_name_formatted}_{self.model_name}_metrics.csv"
        )
        metrics.to_csv(path, index=False)
        return metrics

    def plotResults(self) -> None:
        results = self.getResults()
        results.to_csv(
            os.path.join(self.result_dir, f"{self.appliance_name_formatted}_results.csv"),
            index=False,
        )
        time_values = pd.to_datetime(results["time"])
        for appliance in self.target.appliances:
            plt.figure(figsize=(16, 5))
            plt.plot(time_values, results["aggregate"], label="Aggregate", alpha=0.5)
            plt.plot(time_values, results[f"{appliance}_ground_truth"], label="Ground truth")
            plt.plot(time_values, results[f"{appliance}_prediction"], label="Prediction")
            plt.ylabel("Power (W)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                os.path.join(self.result_dir, f"prediction_plot_{appliance}_{self.model_name}.png"),
                dpi=150,
            )
            plt.close()


Tester = Evaluator
Evaluator.__test__ = False

__all__ = ["Evaluator", "Tester", "METRIC_NAMES", "compute_metrics"]
