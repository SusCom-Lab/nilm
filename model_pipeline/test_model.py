from __future__ import annotations

import json
import os
import re
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, precision_recall_curve, precision_recall_fscore_support
from torch.utils.data import DataLoader

from model_pipeline.classical_data import load_grouped_classical_data
from model_pipeline.contracts import InferenceContext, InferenceData
from model_pipeline.data_protocol import household_id_from_path
from model_pipeline.data_feeder import SlidingWindowDataset, reconstruct_series_from_windows
from model_pipeline.model_registry import instantiate_from_checkpoint, load_checkpoint
from model_pipeline.train_model import compute_metrics


STATUS_RULES_FILE = os.path.join(os.path.dirname(__file__), "appliance_status_rules.json")
STATUS_METRIC_COLUMNS = {
    "Precision": np.nan,
    "Recall": np.nan,
    "F1-score": np.nan,
    "PR-AUC": np.nan,
    "FPR": np.nan,
    "MAE_on": np.nan,
    "MAE_off": np.nan,
    "BMAE": np.nan,
    "status_threshold": np.nan,
    "status_positive_ratio": np.nan,
}
_STATUS_RULES_CACHE = None


def _normalise_appliance_name(name):
    value = str(name or "").strip()
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    value = re.sub(r"[^0-9A-Za-z]+", "_", value).strip("_").lower()
    return value


def _load_status_rules():
    global _STATUS_RULES_CACHE
    if _STATUS_RULES_CACHE is None:
        try:
            with open(STATUS_RULES_FILE, "r", encoding="utf-8") as handle:
                raw_rules = json.load(handle)
        except Exception:
            raw_rules = {}
        _STATUS_RULES_CACHE = {
            _normalise_appliance_name(appliance): dict(rule)
            for appliance, rule in raw_rules.items()
        }
    return _STATUS_RULES_CACHE


def _get_status_rule(appliance_name):
    rules = _load_status_rules()
    normalised = _normalise_appliance_name(appliance_name)
    if normalised in rules:
        return rules[normalised]

    compact = normalised.replace("_", "")
    for rule_name, rule in rules.items():
        if rule_name.replace("_", "") == compact:
            return rule
    return None


def _infer_sample_period_seconds(timestamps):
    try:
        time_values = pd.to_datetime(pd.Series(timestamps), errors="coerce").dropna()
        diffs = time_values.sort_values().diff().dropna().dt.total_seconds()
        diffs = diffs[diffs > 0]
        if not diffs.empty:
            sample_period = float(diffs.median())
            if np.isfinite(sample_period) and sample_period > 0:
                return sample_period
    except Exception:
        pass
    return 1.0


def _duration_to_samples(duration_seconds, sample_period_seconds):
    duration = float(duration_seconds or 0)
    sample_period = float(sample_period_seconds or 1.0)
    if duration <= 0:
        return 0
    if not np.isfinite(sample_period) or sample_period <= 0:
        sample_period = 1.0
    return max(1, int(np.ceil(duration / sample_period)))


def _iter_runs(status):
    if len(status) == 0:
        return

    start = 0
    current = int(status[0])
    for idx in range(1, len(status)):
        value = int(status[idx])
        if value != current:
            yield start, idx, current
            start = idx
            current = value
    yield start, len(status), current


def _apply_duration_rules(status, rule, sample_period_seconds):
    filtered = np.asarray(status, dtype=np.int8).copy()
    min_off_samples = _duration_to_samples(rule.get("min_off_duration", 0), sample_period_seconds)
    min_on_samples = _duration_to_samples(rule.get("min_on_duration", 0), sample_period_seconds)

    if min_off_samples > 0:
        for start, end, value in list(_iter_runs(filtered)):
            if value == 0 and start > 0 and end < len(filtered) and (end - start) < min_off_samples:
                filtered[start:end] = 1

    if min_on_samples > 0:
        for start, end, value in list(_iter_runs(filtered)):
            if value == 1 and (end - start) < min_on_samples:
                filtered[start:end] = 0

    return filtered


def _power_to_status(power, rule, timestamps):
    values = np.asarray(power, dtype=np.float32)
    min_threshold = float(rule["min_threshold"])
    max_threshold = float(rule.get("max_threshold", np.inf))
    initial_status = ((values >= min_threshold) & (values <= max_threshold)).astype(np.int8)
    sample_period = _infer_sample_period_seconds(timestamps)
    return _apply_duration_rules(initial_status, rule, sample_period)


def _prediction_to_status(power, rule, timestamps):
    values = np.asarray(power, dtype=np.float32).copy()
    min_threshold = float(rule["min_threshold"])
    max_threshold = float(rule.get("max_threshold", np.inf))
    values[values < min_threshold] = 0
    initial_status = ((values > min_threshold) & (values <= max_threshold)).astype(np.int8)
    sample_period = _infer_sample_period_seconds(timestamps)
    return _apply_duration_rules(initial_status, rule, sample_period)


def _compute_status_metrics(prediction, ground_truth, timestamps, appliance_name, true_status):
    rule = _get_status_rule(appliance_name)
    metrics = dict(STATUS_METRIC_COLUMNS)
    if rule is None:
        return metrics, None, None, None
    if true_status is None:
        raise ValueError("Status metrics require a 'status' column in the evaluation CSV.")

    y_true = np.asarray(true_status, dtype=np.int8)
    y_pred = _prediction_to_status(prediction, rule, timestamps)
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )
    false_positives = int(((y_pred == 1) & (y_true == 0)).sum())
    true_negatives = int(((y_pred == 0) & (y_true == 0)).sum())
    fpr_denominator = false_positives + true_negatives
    absolute_errors = np.abs(np.asarray(prediction, dtype=np.float32) - np.asarray(ground_truth, dtype=np.float32))
    on_mask = y_true == 1
    off_mask = y_true == 0
    mae_on = float(np.mean(absolute_errors[on_mask])) if bool(on_mask.any()) else np.nan
    mae_off = float(np.mean(absolute_errors[off_mask])) if bool(off_mask.any()) else np.nan
    bmae = float((mae_on + mae_off) / 2.0) if np.isfinite(mae_on) and np.isfinite(mae_off) else np.nan

    metrics.update(
        {
            "Precision": float(precision),
            "Recall": float(recall),
            "F1-score": float(f1_score),
            "FPR": float(false_positives / fpr_denominator) if fpr_denominator else np.nan,
            "MAE_on": mae_on,
            "MAE_off": mae_off,
            "BMAE": bmae,
            "status_threshold": float(rule["min_threshold"]),
            "status_positive_ratio": float(np.mean(y_true)) if len(y_true) else np.nan,
        }
    )

    unique_true = np.unique(y_true)
    if len(unique_true) > 1:
        y_score = np.asarray(prediction, dtype=np.float32)
        metrics["PR-AUC"] = float(average_precision_score(y_true, y_score))
        return metrics, y_true, y_score, rule

    return metrics, y_true, np.asarray(prediction, dtype=np.float32), rule


class Evaluator:
    def __init__(
        self,
        model=None,
        test_loader=None,
        *,
        model_state_dir=None,
        test_csv_dir=None,
        dataset=None,
        result_dir=None,
        batch_size=1000,
        device=None,
        normalisation_params=None,
        timestamps=None,
        aggregate_series=None,
        target_series=None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.MSELoss()
        self.dataset = (dataset or "unknown").lower()

        checkpoint = None
        if model_state_dir is not None:
            checkpoint = load_checkpoint(model_state_dir, map_location=self.device)
            model = instantiate_from_checkpoint(checkpoint, map_location=self.device)
        if model is None:
            raise ValueError("Evaluator requires a model instance or model_state_dir.")

        self.model = model
        self.model_name = getattr(model, "display_name", model.__class__.__name__)
        self.appliance_name_formatted = (checkpoint or {}).get("appliance", "appliance")
        self.joint_mode = bool((checkpoint or {}).get("joint_mode", False))
        self.joint_appliances = list((checkpoint or {}).get("appliances", []))
        checkpoint_normalisation = (checkpoint or {}).get("normalisation_stats")

        self.result_dir = result_dir or os.path.join("result", self.dataset, self.appliance_name_formatted)
        os.makedirs(self.result_dir, exist_ok=True)

        self.batch_size = batch_size
        self.test_csv_dir = test_csv_dir
        self.normalisation_params = normalisation_params or checkpoint_normalisation
        self.raw_timestamps = None if timestamps is None else pd.Series(timestamps).reset_index(drop=True)
        self.raw_aggregate = None if aggregate_series is None else np.asarray(aggregate_series, dtype=np.float32)
        self.raw_target = None if target_series is None else np.asarray(target_series, dtype=np.float32)

        if test_loader is None and test_csv_dir is not None and not self.joint_mode:
            if self.normalisation_params is None:
                raise ValueError(
                    "Evaluator requires training normalisation_params or a checkpoint containing normalisation_stats."
                )
            test_dataset = SlidingWindowDataset(
                [test_csv_dir],
                self.model.get_window_size(),
                target_mode=self.model.get_target_type(),
                output_size=self.model.get_output_size(),
                output_offset=self.model.get_output_offset(),
                normalisation_stats=self.normalisation_params,
            )
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
            self.normalisation_params = test_dataset.getNormalisationParams(test_csv_dir)
            self.window_start_indices = test_dataset.get_window_locations()
        else:
            self.window_start_indices = None

        self.test_loader = test_loader
        if getattr(self.model, "supports_gradient", False):
            self.model.to(self.device)

        self.dt = 0.0
        self.predictions = []
        self.ground_truth = []
        self.aggregate = []
        self.timestamps = pd.Series(dtype=object)
        self.true_status = None
        self.joint_results: dict[str, pd.DataFrame] = {}
        self.joint_metrics = pd.DataFrame()
        self._joint_grouped_data = None

    @staticmethod
    def predict_aggregate_loader(loader, forward_fn, *, device):
        """Run aggregate-only inference without requiring or touching labels."""
        prediction_windows = []
        with torch.no_grad():
            for batch in loader:
                inputs = batch[0] if isinstance(batch, (tuple, list)) else batch
                outputs = forward_fn(inputs.to(device))
                prediction_windows.append(outputs.reshape(-1).detach().cpu().numpy())
        if not prediction_windows:
            raise ValueError("Prediction loader produced no windows.")
        return np.concatenate(prediction_windows).astype(np.float32)

    @staticmethod
    def score_aligned_predictions(
        prediction,
        ground_truth,
        aggregate,
        true_status,
        timestamps,
        *,
        appliance_name,
        gamma=None,
        beta=None,
    ):
        """Score predictions after labels have been explicitly unsealed and aligned."""
        prediction = np.asarray(prediction, dtype=np.float32)
        ground_truth = np.clip(np.asarray(ground_truth, dtype=np.float32), 0.0, None)
        aggregate = np.clip(np.asarray(aggregate, dtype=np.float32), 0.0, None)
        prediction = np.minimum(np.clip(prediction, 0.0, None), aggregate)
        true_status = np.asarray(true_status, dtype=np.int8)
        base_metrics = compute_metrics(prediction, ground_truth)
        status_metrics, _, _, _ = _compute_status_metrics(
            prediction,
            ground_truth,
            timestamps,
            appliance_name,
            true_status,
        )
        time_values = pd.to_datetime(pd.Series(timestamps))
        diffs = time_values.diff().dt.total_seconds().dropna()
        positive_diffs = diffs[diffs > 0]
        sample_period = float(positive_diffs.median()) if not positive_diffs.empty else 1.0
        on_mask = true_status.astype(bool)

        def tensor_norm(value):
            if value is None:
                return 0.0
            return float(torch.linalg.vector_norm(value).item())

        return {
            "MAE": base_metrics["MAE"],
            "MAE-on": status_metrics["MAE_on"],
            "MAE-off": status_metrics["MAE_off"],
            "SAE": base_metrics["SAE"],
            "Precision": status_metrics["Precision"],
            "Recall": status_metrics["Recall"],
            "F1": status_metrics["F1-score"],
            "FPR": status_metrics["FPR"],
            "true_total_energy_Wh": float(np.sum(ground_truth) * sample_period / 3600.0),
            "pred_total_energy_Wh": float(np.sum(prediction) * sample_period / 3600.0),
            "true_ON_mean_power_W": (
                float(np.mean(ground_truth[on_mask])) if bool(on_mask.any()) else np.nan
            ),
            "pred_ON_mean_power_W": (
                float(np.mean(prediction[on_mask])) if bool(on_mask.any()) else np.nan
            ),
            "gamma_norm": tensor_norm(gamma),
            "beta_norm": tensor_norm(beta),
        }

    def _run_batch(self, inputs, targets):
        if getattr(self.model, "supports_gradient", False):
            inputs = inputs.to(self.device)
            targets = self.model.prepare_targets(targets.to(self.device))
            loss_hook = getattr(self.model, "compute_loss", None)
            if callable(loss_hook):
                loss_output = loss_hook(inputs, targets, criterion=self.criterion)
                if isinstance(loss_output, tuple):
                    loss, outputs = loss_output
                    outputs = self.model.prepare_outputs(outputs)
                else:
                    loss = loss_output
                    outputs = self.model.prepare_outputs(self.model(inputs))
            else:
                outputs = self.model.prepare_outputs(self.model(inputs))
                loss = self.criterion(outputs, targets)
            loss = float(loss.item())
            predictions = outputs.detach().cpu().numpy()
            targets_np = targets.detach().cpu().numpy()
            return predictions, targets_np, loss

        inputs_np = inputs.detach().cpu().numpy()
        targets_np = self.model.prepare_targets(targets.detach().cpu().numpy())
        predictions = self.model.prepare_outputs(self.model.disaggregate(inputs_np))
        loss = self.criterion(
            torch.as_tensor(predictions, dtype=torch.float32),
            torch.as_tensor(targets_np, dtype=torch.float32),
        ).item()
        return predictions, targets_np, loss

    def testModel(self):
        if self.joint_mode:
            if not self.test_csv_dir:
                raise ValueError("Joint classical evaluation requires test CSV inputs.")

            csv_paths = self.test_csv_dir if isinstance(self.test_csv_dir, list) else [self.test_csv_dir]
            grouped = load_grouped_classical_data(csv_paths)
            self._joint_grouped_data = grouped

            time_start = time.time()
            self.joint_results = self.model.disaggregate_joint(grouped)
            self.dt = time.time() - time_start

            metric_rows = []
            for group_id, predictions_df in self.joint_results.items():
                group = grouped[group_id]
                for appliance_name in predictions_df.columns:
                    prediction = np.clip(
                        predictions_df[appliance_name].to_numpy(dtype=np.float32),
                        0.0,
                        group.aggregate,
                    )
                    ground_truth = np.clip(group.appliances[appliance_name], 0.0, None)
                    metrics = compute_metrics(prediction, ground_truth)
                    status_metrics, _, _, _ = _compute_status_metrics(
                        prediction,
                        ground_truth,
                        group.time,
                        appliance_name,
                        None,
                    )
                    metric_rows.append(
                        {
                            "group": group_id,
                            "appliance": appliance_name,
                            "model": self.model_name,
                            "dataset": self.dataset,
                            "MAE": metrics["MAE"],
                            "SAE": metrics["SAE"],
                            "inference_time": self.dt,
                            **status_metrics,
                        }
                    )
            self.joint_metrics = pd.DataFrame(metric_rows)
            return

        if self.test_loader is None and not getattr(self.model, "supports_gradient", False):
            raise ValueError("Evaluator requires a test_loader or test_csv_dir.")

        if self.test_csv_dir is not None:
            raw_df = pd.read_csv(self.test_csv_dir, low_memory=False)
            raw_timestamps = raw_df.iloc[:, 0].reset_index(drop=True)
            raw_aggregate = raw_df.iloc[:, 1].astype(float).to_numpy()
            raw_target = raw_df.iloc[:, 2].astype(float).to_numpy()
            if "status" not in raw_df.columns:
                raise ValueError("Evaluation CSV is missing required 'status' column.")
            raw_status = raw_df["status"].astype(float).to_numpy()
        else:
            if self.raw_timestamps is None or self.raw_aggregate is None or self.raw_target is None:
                raise ValueError(
                    "Evaluator requires timestamps, aggregate_series, and target_series when test_csv_dir is not provided."
                )
            raw_timestamps = self.raw_timestamps
            raw_aggregate = self.raw_aggregate
            raw_target = self.raw_target
            raw_status = None
        total_length = len(raw_target)

        if self.normalisation_params is None:
            raise ValueError(
                "Evaluator requires normalisation_params when test_csv_dir is not provided."
            )

        if getattr(self.model, "supports_gradient", False):
            household_id = (
                household_id_from_path(self.test_csv_dir)
                if self.test_csv_dir is not None
                else "unknown"
            )
            segment_ids = (
                raw_df["segment_id"].to_numpy()
                if self.test_csv_dir is not None and "segment_id" in raw_df.columns
                else None
            )
            inference_data = InferenceData(
                timestamps=np.asarray(raw_timestamps),
                aggregate=np.asarray(raw_aggregate, dtype=np.float32),
                household_id=household_id,
                segment_ids=segment_ids,
                source=self.test_csv_dir,
            )
            inference_context = InferenceContext(
                device=self.device,
                batch_size=self.batch_size,
                normalisation_stats=self.normalisation_params,
            )

            time_start = time.time()
            prediction_output = self.model.predict(inference_data, inference_context)
            self.dt = time.time() - time_start
            if len(prediction_output.timestamps) != total_length:
                raise ValueError(
                    "Model predict() must return the complete protocol test timeline."
                )
            if not pd.Index(prediction_output.timestamps).equals(pd.Index(raw_timestamps)):
                raise ValueError("Model prediction timestamps do not match the test timeline.")

            valid_mask = (
                np.ones(total_length, dtype=bool)
                if prediction_output.valid_mask is None
                else np.asarray(prediction_output.valid_mask, dtype=bool)
            )
            self.timestamps = raw_timestamps[valid_mask].reset_index(drop=True)
            self.aggregate = np.clip(raw_aggregate[valid_mask], 0.0, None).tolist()
            self.ground_truth = np.clip(raw_target[valid_mask], 0.0, None).tolist()
            self.true_status = (
                None
                if raw_status is None
                else raw_status[valid_mask].astype(np.int8).tolist()
            )
            predictions = np.asarray(prediction_output.power, dtype=np.float32)[valid_mask]
            predictions = np.minimum(
                np.clip(predictions, 0.0, None),
                np.asarray(self.aggregate, dtype=np.float32),
            )
            self.predictions = predictions.tolist()
            test_loss = self.criterion(
                torch.as_tensor(predictions),
                torch.as_tensor(self.ground_truth),
            ).item()
            print(f"Test Loss: {test_loss}")
            return

        appliance_mean = self.normalisation_params["appliance_mean"]
        appliance_std = self.normalisation_params["appliance_std"]

        prediction_windows = []
        test_loss = 0.0
        time_start = time.time()

        if getattr(self.model, "supports_gradient", False):
            self.model.eval()

        with torch.no_grad() if getattr(self.model, "supports_gradient", False) else _nullcontext():
            for inputs, targets in self.test_loader:
                predictions, _, loss = self._run_batch(inputs, targets)
                prediction_windows.append(predictions)
                test_loss += loss

        self.dt = time.time() - time_start
        prediction_windows = np.concatenate(prediction_windows, axis=0)
        prediction_windows = (prediction_windows * appliance_std) + appliance_mean

        reconstructed, coverage = reconstruct_series_from_windows(
            prediction_windows,
            total_length,
            target_mode=self.model.get_target_type(),
            output_offset=self.model.get_output_offset(),
            output_size=self.model.get_output_size(),
            window_start_indices=self.window_start_indices,
        )
        valid_mask = coverage > 0

        self.timestamps = raw_timestamps[valid_mask].reset_index(drop=True)
        self.aggregate = np.clip(raw_aggregate[valid_mask], 0.0, None).tolist()
        self.ground_truth = np.clip(raw_target[valid_mask], 0.0, None).tolist()
        self.true_status = None if raw_status is None else raw_status[valid_mask].astype(np.int8).tolist()
        predictions = np.clip(reconstructed[valid_mask], 0.0, None)
        predictions = np.minimum(predictions, np.asarray(self.aggregate, dtype=np.float32))
        self.predictions = predictions.tolist()

        test_loss /= max(1, len(self.test_loader))
        print(f"Test Loss: {test_loss}")

    def evaluate(self):
        self.testModel()
        return self.getMetrics()

    def getResults(self):
        if self.joint_mode:
            return dict(self.joint_results)
        data = {
            "time": self.timestamps,
            "aggregate": self.aggregate,
            "prediction": self.predictions,
            "ground truth": self.ground_truth,
        }
        if self.true_status is not None:
            data["status"] = self.true_status
        return pd.DataFrame(data)

    def getMetrics(self):
        if self.joint_mode:
            return self.joint_metrics.copy()
        metrics = compute_metrics(self.predictions, self.ground_truth)
        return metrics["MAE"], metrics["SAE"], self.dt

    def saveMetrics(self):
        if self.joint_mode:
            metrics_filename = f"{self.model_name}_metrics.csv"
            metrics_path = os.path.join(self.result_dir, metrics_filename)
            self.joint_metrics.to_csv(metrics_path, index=False)
            print(f"Metrics saved to {metrics_path}")
            return self.joint_metrics

        mae, sae, dt = self.getMetrics()
        status_metrics, _, _, _ = _compute_status_metrics(
            self.predictions,
            self.ground_truth,
            self.timestamps,
            self.appliance_name_formatted,
            self.true_status,
        )
        metrics_df = pd.DataFrame(
            {
                "appliance": [self.appliance_name_formatted],
                "model": [self.model_name],
                "dataset": [self.dataset],
                "MAE": [mae],
                "SAE": [sae],
                "inference_time": [dt],
                **{name: [value] for name, value in status_metrics.items()},
            }
        )
        metrics_filename = f"{self.appliance_name_formatted}_{self.model_name}_metrics.csv"
        metrics_path = os.path.join(self.result_dir, metrics_filename)
        metrics_df.to_csv(metrics_path, index=False)
        print(f"Metrics saved to {metrics_path}")
        return mae, sae, dt

    def _get_zoom_window_file(self):
        return os.path.join(self.result_dir, f"{self.appliance_name_formatted}_zoom_window.json")

    def _load_saved_zoom_window(self):
        zoom_file = self._get_zoom_window_file()
        if os.path.exists(zoom_file):
            try:
                with open(zoom_file, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
                return data.get("start"), data.get("end")
            except Exception:
                pass
        return None, None

    def _save_zoom_window(self, start_idx, end_idx):
        with open(self._get_zoom_window_file(), "w", encoding="utf-8") as handle:
            json.dump({"start": int(start_idx), "end": int(end_idx)}, handle)

    def _find_active_region(self, min_peaks=2, window_size=100):
        try:
            from scipy.signal import find_peaks
        except Exception:
            nonzero = np.flatnonzero(np.asarray(self.ground_truth, dtype=np.float32) > 0)
            if nonzero.size == 0:
                return None
            start_idx = int(nonzero[0])
            end_idx = min(len(self.ground_truth), start_idx + window_size)
            return start_idx, end_idx

        ground_truth = np.asarray(self.ground_truth, dtype=np.float32)
        if ground_truth.size == 0:
            return None

        threshold = np.max(ground_truth) * 0.4
        best_start = None
        best_end = None
        best_peak = 0.0

        for start_idx in range(0, max(1, len(ground_truth) - window_size), max(1, window_size // 2)):
            end_idx = min(start_idx + window_size, len(ground_truth))
            segment = ground_truth[start_idx:end_idx]
            if np.max(segment) < threshold:
                continue

            peaks, properties = find_peaks(
                segment,
                height=threshold * 0.5,
                distance=10,
                prominence=threshold * 0.3,
            )
            peak_heights = properties.get("peak_heights", [])
            max_peak = float(np.max(peak_heights)) if len(peak_heights) > 0 else 0.0
            if len(peaks) >= min_peaks and max_peak > best_peak:
                best_peak = max_peak
                best_start = start_idx
                best_end = end_idx

        if best_start is None:
            return None
        return best_start, best_end

    def _get_zoom_window(self):
        saved_start, saved_end = self._load_saved_zoom_window()
        if saved_start is not None and saved_end is not None and saved_end <= len(self.ground_truth):
            return saved_start, saved_end

        active_region = self._find_active_region()
        if active_region is not None:
            self._save_zoom_window(*active_region)
            return active_region
        return 0, min(10, len(self.ground_truth))

    def plotResults(self):
        if self.joint_mode:
            if not self.joint_results:
                return

            grouped = self._joint_grouped_data
            if grouped is None:
                csv_paths = self.test_csv_dir if isinstance(self.test_csv_dir, list) else [self.test_csv_dir]
                grouped = load_grouped_classical_data(csv_paths)

            for group_id, predictions_df in self.joint_results.items():
                group = grouped[group_id]
                for appliance_name in predictions_df.columns:
                    result_df = pd.DataFrame(
                        {
                            "time": group.time.reset_index(drop=True),
                            "aggregate": np.clip(group.aggregate, 0.0, None),
                            "prediction": np.clip(
                                predictions_df[appliance_name].to_numpy(dtype=np.float32),
                                0.0,
                                group.aggregate,
                            ),
                            "ground truth": np.clip(group.appliances[appliance_name], 0.0, None),
                        }
                    )
                    safe_appliance = appliance_name.replace(" ", "_")
                    result_path = os.path.join(self.result_dir, f"{safe_appliance}_results.csv")
                    result_df.to_csv(result_path, index=False)
                    print(f"Results CSV saved to {result_path}")

                    plot_df = result_df.copy()
                    plot_df["time"] = pd.to_datetime(plot_df["time"])
                    plt.figure(figsize=(30, 6))
                    plt.plot(plot_df["time"], plot_df["aggregate"], label="Aggregate", alpha=0.7)
                    plt.plot(plot_df["time"], plot_df["ground truth"], label="Ground Truth", alpha=0.7)
                    plt.plot(plot_df["time"], plot_df["prediction"], label="Prediction", alpha=0.7)
                    plt.title(f"Prediction Plot for {safe_appliance} using {self.model_name}")
                    plt.xlabel("Timestamp")
                    plt.ylabel("Power (Watts)")
                    plt.legend()
                    plt.xticks(rotation=45)
                    plt.tight_layout()

                    plot_filename = f"prediction_plot_{safe_appliance}_{self.model_name}.png"
                    plot_path = os.path.join(self.result_dir, plot_filename)
                    plt.savefig(plot_path, bbox_inches="tight", dpi=300)
                    plt.close()
                    print(f"Plot saved to {plot_path}")

            self.saveMetrics()
            return

        results_df = self.getResults()
        if results_df.empty:
            return

        results_df["time"] = pd.to_datetime(results_df["time"])
        plot_df = results_df.copy()
        time_diffs = plot_df["time"].diff()
        if len(plot_df) > 1:
            reference_gap = time_diffs.dropna().min()
            if pd.notna(reference_gap):
                gap_threshold = reference_gap * 2
                gap_mask = time_diffs > gap_threshold
                plot_df.loc[gap_mask, ["aggregate", "ground truth", "prediction"]] = np.nan

        plt.figure(figsize=(30, 6))
        plt.plot(plot_df["time"], plot_df["aggregate"], label="Aggregate", alpha=0.7)
        plt.plot(plot_df["time"], plot_df["ground truth"], label="Ground Truth", alpha=0.7)
        plt.plot(plot_df["time"], plot_df["prediction"], label="Prediction", alpha=0.7)
        plt.title(f"Prediction Plot for {self.appliance_name_formatted} using {self.model_name}")
        plt.xlabel("Timestamp")
        plt.ylabel("Power (Watts)")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()

        mae, sae, dt = self.getMetrics()
        plt.figtext(0.15, 0.01, f"MAE: {mae:.2f} Watts, SAE: {sae:.2f}, Inference Time: {dt:.2f} seconds", ha="left", fontsize=12)

        plot_filename = f"prediction_plot_{self.appliance_name_formatted}_{self.model_name}.png"
        plot_path = os.path.join(self.result_dir, plot_filename)
        plt.savefig(plot_path, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"Plot saved to {plot_path}")

        zoom_start, zoom_end = self._get_zoom_window()
        zoom_ground_truth = results_df["ground truth"].iloc[zoom_start:zoom_end]
        zoom_prediction = results_df["prediction"].iloc[zoom_start:zoom_end]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(range(len(zoom_ground_truth)), zoom_ground_truth, color="#1f77b4", label="Ground truth", linewidth=1.8)
        ax.plot(range(len(zoom_prediction)), zoom_prediction, color="#ff7f0e", label=self.model_name, linewidth=1.8)
        ax.set_ylabel("Power (W)")
        ax.legend(loc="upper left", frameon=True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xticks([])
        plt.tight_layout()

        zoom_plot_filename = f"zoomed_plot_{self.appliance_name_formatted}_{self.model_name}.png"
        zoom_plot_path = os.path.join(self.result_dir, zoom_plot_filename)
        plt.savefig(zoom_plot_path, bbox_inches="tight", dpi=150)
        plt.close()
        print(f"Zoomed plot saved to {zoom_plot_path}")

        results_filename = f"{self.appliance_name_formatted}_results.csv"
        results_path = os.path.join(self.result_dir, results_filename)
        results_df.to_csv(results_path, index=False)
        print(f"Results CSV saved to {results_path}")
        self._plot_pr_curve(results_df)
        self.saveMetrics()

    def _plot_pr_curve(self, results_df):
        true_status = results_df["status"].to_numpy(dtype=np.int8) if "status" in results_df.columns else self.true_status
        status_metrics, y_true, y_score, _ = _compute_status_metrics(
            results_df["prediction"].to_numpy(dtype=np.float32),
            results_df["ground truth"].to_numpy(dtype=np.float32),
            results_df["time"],
            self.appliance_name_formatted,
            true_status,
        )
        if y_true is None or y_score is None:
            print(f"PR curve skipped: no status rule for {self.appliance_name_formatted}.")
            return
        if len(np.unique(y_true)) < 2:
            print(f"PR curve skipped: ground truth status has only one class for {self.appliance_name_formatted}.")
            return

        precision, recall, _ = precision_recall_curve(y_true, y_score)
        pr_auc = status_metrics["PR-AUC"]

        plt.figure(figsize=(6, 5))
        plt.plot(recall, precision, label=f"PR-AUC: {pr_auc:.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"PR Curve for {self.appliance_name_formatted} using {self.model_name}")
        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.05)
        plt.legend(loc="lower left")
        plt.tight_layout()

        plot_filename = f"pr_curve_{self.appliance_name_formatted}_{self.model_name}.png"
        plot_path = os.path.join(self.result_dir, plot_filename)
        plt.savefig(plot_path, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"PR curve saved to {plot_path}")


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, exc_tb):
        return False


Tester = Evaluator
