from model_pipeline.data_feeder import SlidingWindowDataset
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from model_pipeline.seq2Point_factory import Seq2PointFactory
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import re
import json
import time 

class Tester:
    def __init__(self, model_state_dir, test_csv_dir, dataset=None, result_dir=None):
        """
        Tester class for testing the model
        model_name (str): Name of the model to test.
        model_state_dir (str): Directory to load the model state from.
        test_csv_dir (str): Directory to load the test CSV from.
        dataset (str): Name of the dataset (e.g., 'redd', 'ukdale').
        result_dir (str): Directory to save results (default: result/{dataset}/{appliance}/).
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.criterion = nn.MSELoss()
        
        if dataset is None:
            self.dataset = "unknown"
        else:
            self.dataset = dataset.lower()

        checkpoint = torch.load(model_state_dir, map_location=self.device)
        self.model_name = checkpoint['model_name']
        window_length = checkpoint['window_length']
        self.model = Seq2PointFactory.createModel(self.model_name, window_length)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)

        self.appliance_name_formatted = checkpoint['appliance']

        if result_dir is None:
            self.result_dir = os.path.join("result", self.dataset, self.appliance_name_formatted)
        else:
            self.result_dir = os.path.join(result_dir, self.dataset, self.appliance_name_formatted)
        os.makedirs(self.result_dir, exist_ok=True)

        self.batch_size = 1000
        self.offset = int((0.5 * window_length) - 1)
        test_dataset = SlidingWindowDataset([test_csv_dir], self.model.getWindowSize())
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        normalisation_params = test_dataset.getNormalisationParams(test_csv_dir)
        self.aggregate_mean = normalisation_params["aggregate_mean"]
        self.aggregate_std = normalisation_params["aggregate_std"]
        self.appliance_mean = normalisation_params["appliance_mean"]
        self.appliance_std = normalisation_params["appliance_std"]

        test_df = pd.read_csv(test_csv_dir, low_memory=False)
        self.dt = 0
        self.timestamps = test_df["time"].iloc[self.offset:-self.offset].reset_index(drop=True)
        self.predictions = []
        self.ground_truth = []
        self.aggregate = []

    def testModel(self):
        """
        Test the model on the test dataset and collect predictions.
        """
        self.model.eval()
        test_loss = 0
        time_start = time.time()
        with torch.no_grad():
                for inputs, targets in self.test_loader:
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs.squeeze(-1), targets)
                        test_loss += loss.item()

                        denormalised_outputs = outputs.squeeze(-1) * self.appliance_std + self.appliance_mean
                        denormalised_targets = targets * self.appliance_std + self.appliance_mean
                        denormalised_inputs = inputs * self.aggregate_std + self.aggregate_mean

                        self.predictions.extend(denormalised_outputs.cpu().numpy().flatten())
                        self.ground_truth.extend(denormalised_targets.cpu().numpy().flatten())
                        self.aggregate.extend(denormalised_inputs[:, self.offset].cpu().numpy().flatten())


        trim_length = len(self.predictions)
        self.timestamps = self.timestamps[:trim_length]

        self.predictions = [max(0, pred) for pred in self.predictions]
        self.predictions = [min(pred, agg) for pred, agg in zip(self.predictions, self.aggregate)]
        self.ground_truth = [max(0, gt) for gt in self.ground_truth]
        self.aggregate = [max(0, agg) for agg in self.aggregate]

        test_loss /= len(self.test_loader)
        time_end = time.time()
        self.dt = time_end - time_start
        print(f"Test Loss: {test_loss}")

    
    def getResults(self):
        """
        Return the results of the test as a pandas dataframe
        """
        results_df = pd.DataFrame({
                "time": self.timestamps,
                "aggregate": self.aggregate,
                "prediction": self.predictions,
                "ground truth": self.ground_truth
        })
        return results_df

    def getMetrics(self):
        """
        Calculate the metrics for the test.
        Also return the time taken for disaggreation.
        """
        predictions = np.array(self.predictions)
        ground_truth = np.array(self.ground_truth)
        
        MAE = np.mean(np.abs(predictions - ground_truth))
        SAE = abs(sum(predictions) - sum(ground_truth)) / sum(ground_truth)

        return MAE, SAE, self.dt

    def saveMetrics(self):
        """
        Save metrics (MAE, SAE) to a CSV file.
        """
        MAE, SAE, dt = self.getMetrics()
        metrics_df = pd.DataFrame({
            "appliance": [self.appliance_name_formatted],
            "model": [self.model_name],
            "dataset": [self.dataset],
            "MAE": [MAE],
            "SAE": [SAE],
            "inference_time": [dt]
        })
        metrics_filename = f"{self.appliance_name_formatted}_{self.model_name}_metrics.csv"
        metrics_path = os.path.join(self.result_dir, metrics_filename)
        metrics_df.to_csv(metrics_path, index=False)
        print(f"Metrics saved to {metrics_path}")
        return MAE, SAE, dt

    def _get_zoom_window_file(self):
        """
        Get the file path for storing zoom window indices.
        """
        return os.path.join(self.result_dir, f"{self.appliance_name_formatted}_zoom_window.json")

    def _load_saved_zoom_window(self):
        """
        Load previously saved zoom window indices.
        Returns None if file doesn't exist.
        """
        zoom_file = self._get_zoom_window_file()
        if os.path.exists(zoom_file):
            try:
                with open(zoom_file, 'r') as f:
                    data = json.load(f)
                    return data.get('start'), data.get('end')
            except:
                pass
        return None, None

    def _save_zoom_window(self, start_idx, end_idx):
        """
        Save zoom window indices for consistent plotting across different models.
        """
        zoom_file = self._get_zoom_window_file()
        with open(zoom_file, 'w') as f:
            json.dump({'start': start_idx, 'end': end_idx}, f)

    def _find_active_region(self, min_peaks=2, window_size=100):
        """
        Find a region with 2-3 large wave peaks where ground truth has significant values.
        
        Args:
            min_peaks: Minimum number of peaks to find (default 2)
            window_size: Size of the window to search for peaks (default 100)
        
        Returns:
            tuple: (start_index, end_index) or None if no suitable region found
        """
        from scipy.signal import find_peaks
        
        ground_truth = np.array(self.ground_truth)
        data_length = len(ground_truth)
        
        threshold = np.max(ground_truth) * 0.4
        
        best_start = None
        best_end = None
        best_max_peak = 0
        
        step = window_size // 2
        
        for start in range(0, data_length - window_size, step):
            end = min(start + window_size, data_length)
            
            gt_segment = ground_truth[start:end]
            
            if np.max(gt_segment) < threshold:
                continue
            
            peaks, properties = find_peaks(gt_segment, height=threshold * 0.5, distance=10, prominence=threshold * 0.3)
            peak_count = len(peaks)
            peak_heights = properties['peak_heights']
            max_peak_height = np.max(peak_heights) if len(peak_heights) > 0 else 0
            
            if peak_count >= min_peaks and max_peak_height > best_max_peak:
                best_max_peak = max_peak_height
                best_start = start
                best_end = end
        
        if best_start is not None:
            return best_start, best_end
        
        return None

    def _get_zoom_window(self):
        """
        Get zoom window indices for plotting.
        First checks if a window was previously saved (for consistency across models).
        If not, finds a new window where both ground truth and prediction are > 0.
        
        Returns:
            tuple: (start_index, end_index)
        """
        saved_start, saved_end = self._load_saved_zoom_window()
        
        if saved_start is not None and saved_end is not None:
            data_length = len(self.ground_truth)
            if saved_start < data_length and saved_end <= data_length:
                return saved_start, saved_end
        
        active_region = self._find_active_region()
        
        if active_region is not None:
            self._save_zoom_window(active_region[0], active_region[1])
            return active_region
        
        return 0, min(10, len(self.ground_truth))

    def plotResults(self):
        """
        Plot the results of the test and save to result directory.
        """
        results_df = pd.DataFrame({
                "time": self.timestamps,
                "aggregate": self.aggregate,
                "prediction": self.predictions,
                "ground truth": self.ground_truth
        })
        
        plt.figure(figsize=(30, 6))
        results_df["time"] = pd.to_datetime(results_df["time"])
        plt.plot(results_df["time"], results_df["aggregate"], label="Aggregate", alpha=0.7)
        plt.plot(results_df["time"], results_df["ground truth"], label="Ground Truth", alpha=0.7)
        plt.plot(results_df["time"], results_df["prediction"], label="Prediction", alpha=0.7)
        plt.title(f"Prediction Plot for {self.appliance_name_formatted} using {self.model_name}")
        plt.xlabel("Timestamp")
        plt.ylabel("Power (Watts)")
        plt.legend()
        plt.title("Aggregate, Ground Truth, and Prediction Comparison")
        plt.xticks(rotation=45)
        plt.tight_layout()

        mae, sae, dt = self.getMetrics()
        plt.figtext(0.15, 0.01, f"MAE: {mae:.2f} Watts, SAE: {sae:.2f}, Inference Time: {dt:.2f} seconds", ha="left", fontsize=12)
        
        plot_filename = f"prediction_plot_{self.appliance_name_formatted}_{self.model_name}.png"
        plot_path = os.path.join(self.result_dir, plot_filename)
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Plot saved to {plot_path}")
        
        zoom_start, zoom_end = self._get_zoom_window()
        zoom_time = results_df["time"].iloc[zoom_start:zoom_end]
        zoom_ground_truth = results_df["ground truth"].iloc[zoom_start:zoom_end]
        zoom_prediction = results_df["prediction"].iloc[zoom_start:zoom_end]
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(range(len(zoom_ground_truth)), zoom_ground_truth, color='#1f77b4', label='Ground truth', linewidth=1.8)
        ax.plot(range(len(zoom_prediction)), zoom_prediction, color='#ff7f0e', label='Seq2point', linewidth=1.8)
        
        ax.set_xlabel('')
        ax.set_ylabel('Power (W)')
        ax.legend(loc='upper left', frameon=True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xticks([])
        
        plt.tight_layout()
        
        zoom_plot_filename = f"zoomed_plot_{self.appliance_name_formatted}_{self.model_name}.png"
        zoom_plot_path = os.path.join(self.result_dir, zoom_plot_filename)
        plt.savefig(zoom_plot_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Zoomed plot saved to {zoom_plot_path}")
        
        results_filename = f"{self.appliance_name_formatted}_results.csv"
        results_path = os.path.join(self.result_dir, results_filename)
        results_df.to_csv(results_path, index=False)
        print(f"Results CSV saved to {results_path}")
        
        self.saveMetrics()
