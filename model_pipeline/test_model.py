from __future__ import annotations

import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model_pipeline.classical_data import load_grouped_classical_data
from model_pipeline.data_feeder import SlidingWindowDataset, reconstruct_series_from_windows
from model_pipeline.model_registry import instantiate_from_checkpoint, load_checkpoint
from model_pipeline.train_model import compute_metrics


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

        self.test_loader = test_loader
        if getattr(self.model, "supports_gradient", False):
            self.model.to(self.device)

        self.dt = 0.0
        self.predictions = []
        self.ground_truth = []
        self.aggregate = []
        self.timestamps = pd.Series(dtype=object)
        self.joint_results: dict[str, pd.DataFrame] = {}
        self.joint_metrics = pd.DataFrame()
        self._joint_grouped_data = None

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
                    metric_rows.append(
                        {
                            "group": group_id,
                            "appliance": appliance_name,
                            "model": self.model_name,
                            "dataset": self.dataset,
                            "MAE": metrics["MAE"],
                            "SAE": metrics["SAE"],
                            "inference_time": self.dt,
                        }
                    )
            self.joint_metrics = pd.DataFrame(metric_rows)
            return

        if self.test_loader is None:
            raise ValueError("Evaluator requires a test_loader or test_csv_dir.")

        if self.test_csv_dir is not None:
            raw_df = pd.read_csv(self.test_csv_dir, low_memory=False)
            raw_timestamps = raw_df.iloc[:, 0].reset_index(drop=True)
            raw_aggregate = raw_df.iloc[:, 1].astype(float).to_numpy()
            raw_target = raw_df.iloc[:, 2].astype(float).to_numpy()
        else:
            if self.raw_timestamps is None or self.raw_aggregate is None or self.raw_target is None:
                raise ValueError(
                    "Evaluator requires timestamps, aggregate_series, and target_series when test_csv_dir is not provided."
                )
            raw_timestamps = self.raw_timestamps
            raw_aggregate = self.raw_aggregate
            raw_target = self.raw_target
        total_length = len(raw_target)

        if self.normalisation_params is None:
            raise ValueError(
                "Evaluator requires normalisation_params when test_csv_dir is not provided."
            )

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
        )
        valid_mask = coverage > 0

        self.timestamps = raw_timestamps[valid_mask].reset_index(drop=True)
        self.aggregate = np.clip(raw_aggregate[valid_mask], 0.0, None).tolist()
        self.ground_truth = np.clip(raw_target[valid_mask], 0.0, None).tolist()
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
        return pd.DataFrame(
            {
                "time": self.timestamps,
                "aggregate": self.aggregate,
                "prediction": self.predictions,
                "ground truth": self.ground_truth,
            }
        )

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
        metrics_df = pd.DataFrame(
            {
                "appliance": [self.appliance_name_formatted],
                "model": [self.model_name],
                "dataset": [self.dataset],
                "MAE": [mae],
                "SAE": [sae],
                "inference_time": [dt],
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
        plt.figure(figsize=(30, 6))
        plt.plot(results_df["time"], results_df["aggregate"], label="Aggregate", alpha=0.7)
        plt.plot(results_df["time"], results_df["ground truth"], label="Ground Truth", alpha=0.7)
        plt.plot(results_df["time"], results_df["prediction"], label="Prediction", alpha=0.7)
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
        self.saveMetrics()


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, exc_tb):
        return False


Tester = Evaluator
