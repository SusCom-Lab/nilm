"""
DSC baseline adapted to joint multi-appliance disaggregation.

Official source: https://github.com/nilmtk/nilmtk-contrib/blob/14efd545e09d836e02159b23444350af92c6ee70/nilmtk_contrib/torch/dsc.py
Local adaptation: joint-house repository plugin interface.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

import torch

from model_pipeline.api import FitResult, InferenceContext, PredictionOutput, TrainingContext
from model_pipeline.model_registry import register_model
from model_pipeline.models.classical.afhmm import (
    ClassicalGroup,
    ClassicalInferenceGroup,
    _deserialise_value,
    _serialise_value,
    inference_groups,
    supervised_groups,
)


@register_model("dsc", aliases=("DSC",), display_name="DSC")
class DiscriminativeSparseCoding:
    display_name = "DSC"
    model_family = "probabilistic"
    target_type = "sequence"
    supports_gradient = False
    is_joint_classical = True
    default_num_epochs = 1
    official_batch_size = 1

    def __init__(
        self,
        *,
        window_size: int = 120,
        learning_rate: float = 1e-9,
        iterations: int = 3000,
        sparsity_coef: float = 20,
        n_components: int = 10,
    ):
        if (window_size, learning_rate, iterations, sparsity_coef, n_components) != (
            120, 1e-9, 3000, 20, 10
        ):
            raise ValueError("DSC uses the fixed official default configuration.")
        self.window_size = window_size
        self._config = {
            "window_size": window_size,
            "learning_rate": learning_rate,
            "iterations": iterations,
            "sparsity_coef": sparsity_coef,
            "n_components": n_components,
        }
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.sparsity_coef = sparsity_coef
        self.n_components = n_components
        self.appliance_order: list[str] = []
        self.dictionary_components: dict[str, np.ndarray] = {}
        self.reconstruction_bases: np.ndarray | None = None
        self.disaggregation_bases: np.ndarray | None = None
        self.component_slices: dict[str, tuple[int, int]] = {}

    def fit(self, train_data, validation_data, context: TrainingContext) -> FitResult:
        del validation_data
        self.fit_joint(supervised_groups(train_data))
        validation = context.validate_candidate(epoch=1, model=self)
        return FitResult(
            history=[{"epoch": 1, "validation_mse": validation.mse}],
            best_epoch=1 if validation.improved else None,
            best_validation_mse=validation.mse,
            checkpoint_path=validation.checkpoint_path,
        )

    def _reshape_power(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        if values.size % self.window_size != 0:
            extra_values = self.window_size - (values.size % self.window_size)
            values = np.concatenate([values, np.zeros(extra_values, dtype=np.float32)])
        return values.reshape((-1, self.window_size)).T

    def _learn_dictionary(self, appliance_main: np.ndarray):
        from sklearn.decomposition import MiniBatchDictionaryLearning

        model = MiniBatchDictionaryLearning(
            n_components=self.n_components,
            fit_algorithm="cd",
            positive_code=True,
            positive_dict=True,
            transform_algorithm="lasso_lars",
            alpha=self.sparsity_coef,
        )
        model.fit(appliance_main.T)
        return model

    def _discriminative_training(
        self,
        optimal_activations: np.ndarray,
        initial_bases: np.ndarray,
        total_power: np.ndarray,
    ) -> np.ndarray:
        from sklearn.decomposition import SparseCoder

        predicted_bases = np.copy(initial_bases)
        optimal_activations = np.copy(optimal_activations)
        alpha = self.learning_rate
        least_error = float("inf")
        best_bases = np.copy(initial_bases)

        validation_size = max(1, int(total_power.shape[1] * 0.20))
        train_power = total_power[:, :-validation_size]
        val_power = total_power[:, -validation_size:]
        train_optimal_a = optimal_activations[:, :-validation_size]
        val_optimal_a = optimal_activations[:, -validation_size:]

        for _ in range(self.iterations):
            train_coder = SparseCoder(
                dictionary=predicted_bases.T,
                positive_code=True,
                transform_algorithm="lasso_lars",
                transform_alpha=self.sparsity_coef,
            )
            train_predicted_a = train_coder.transform(train_power.T).T

            val_coder = SparseCoder(
                dictionary=predicted_bases.T,
                positive_code=True,
                transform_algorithm="lasso_lars",
                transform_alpha=self.sparsity_coef,
            )
            val_predicted_a = val_coder.transform(val_power.T).T
            error = float(np.mean(np.abs(val_predicted_a - val_optimal_a)))

            if error < least_error:
                least_error = error
                best_bases = np.copy(predicted_bases)

            term_1 = (train_power - predicted_bases @ train_predicted_a) @ train_predicted_a.T
            term_2 = (train_power - predicted_bases @ train_optimal_a) @ train_optimal_a.T
            predicted_bases = predicted_bases - alpha * (term_1 - term_2)
            predicted_bases = np.where(predicted_bases > 0, predicted_bases, 0)
            norms = np.linalg.norm(predicted_bases.T, axis=1).reshape((-1, 1))
            norms = np.where(norms < 1e-8, 1.0, norms)
            predicted_bases = (predicted_bases.T / norms).T

        return best_bases

    def fit_joint(self, grouped_train_data: dict[str, ClassicalGroup]) -> None:
        if not grouped_train_data:
            raise ValueError("DiscriminativeSparseCoding requires non-empty grouped training data.")

        appliance_names = sorted(next(iter(grouped_train_data.values())).appliances.keys())
        self.appliance_order = appliance_names

        aggregate_series = []
        appliance_series: dict[str, list[np.ndarray]] = {name: [] for name in appliance_names}
        for group in grouped_train_data.values():
            if sorted(group.appliances.keys()) != appliance_names:
                raise ValueError("All grouped training entries must contain the same appliance set.")
            aggregate_series.append(group.aggregate)
            for appliance_name in appliance_names:
                appliance_series[appliance_name].append(group.appliances[appliance_name])

        total_power = self._reshape_power(np.concatenate(aggregate_series, axis=0))
        concatenated_bases = []
        concatenated_activations = []
        self.dictionary_components = {}
        self.component_slices = {}

        current_start = 0
        for appliance_name in appliance_names:
            reshaped = self._reshape_power(np.concatenate(appliance_series[appliance_name], axis=0))
            dictionary_model = self._learn_dictionary(reshaped)
            bases = dictionary_model.components_.T.astype(np.float32)
            activations = dictionary_model.transform(reshaped.T).T.astype(np.float32)
            self.dictionary_components[appliance_name] = dictionary_model.components_.astype(np.float32)
            concatenated_bases.append(bases)
            concatenated_activations.append(activations)
            self.component_slices[appliance_name] = (current_start, current_start + self.n_components)
            current_start += self.n_components

        all_bases = np.concatenate(concatenated_bases, axis=1)
        all_activations = np.concatenate(concatenated_activations, axis=0)
        optimal_bases = self._discriminative_training(all_activations, all_bases, total_power)

        self.reconstruction_bases = all_bases.astype(np.float32)
        self.disaggregation_bases = optimal_bases.astype(np.float32)

    def disaggregate_joint(
        self,
        grouped_test_data: dict[str, ClassicalInferenceGroup],
    ) -> dict[str, pd.DataFrame]:
        if self.disaggregation_bases is None or self.reconstruction_bases is None:
            raise RuntimeError("DiscriminativeSparseCoding must be fitted before disaggregation.")

        from sklearn.decomposition import SparseCoder

        coder = SparseCoder(
            dictionary=self.disaggregation_bases.T,
            positive_code=True,
            transform_algorithm="lasso_lars",
            transform_alpha=self.sparsity_coef,
        )

        predictions: dict[str, pd.DataFrame] = {}
        for group_id, group in grouped_test_data.items():
            reshaped = self._reshape_power(group.aggregate)
            predicted_activations = coder.transform(reshaped.T).T
            outputs = {}
            for appliance_name in self.appliance_order:
                start, end = self.component_slices[appliance_name]
                predicted_usage = (
                    self.reconstruction_bases[:, start:end] @ predicted_activations[start:end, :]
                ).T.reshape(-1)
                predicted_usage = predicted_usage[: len(group.aggregate)]
                outputs[appliance_name] = np.clip(predicted_usage, 0.0, group.aggregate)
            predictions[group_id] = pd.DataFrame(outputs, dtype="float32")
        return predictions

    def predict(self, inference_data, context: InferenceContext) -> PredictionOutput:
        del context
        predictions = self.disaggregate_joint(inference_groups(inference_data))
        timestamps, households, powers = [], [], []
        for series in inference_data.series:
            frame = predictions[series.household_id]
            timestamps.append(series.timestamps)
            households.append(np.repeat(series.household_id, len(series.timestamps)))
            powers.append(frame[list(series.appliances)].to_numpy(dtype=np.float32))
        return PredictionOutput(
            timestamps=np.concatenate(timestamps),
            power=np.concatenate(powers),
            appliances=inference_data.series[0].appliances,
            household_ids=np.concatenate(households),
        )

    def save(self, path, *, metadata=None) -> None:
        torch.save(
            {
                "model_key": self._registry_key,
                "init_kwargs": dict(self._config),
                "model_state": self.get_state(),
                "metadata": dict(metadata or {}),
            }, path,
        )

    def load(self, path, device="cpu") -> None:
        try:
            checkpoint = torch.load(path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location=device)
        self.set_state(checkpoint["model_state"])

    def get_state(self) -> dict[str, Any]:
        return _serialise_value(
            {
                "appliance_order": self.appliance_order,
                "dictionary_components": self.dictionary_components,
                "reconstruction_bases": self.reconstruction_bases,
                "disaggregation_bases": self.disaggregation_bases,
                "component_slices": self.component_slices,
            }
        )

    def set_state(self, state: dict[str, Any]) -> None:
        restored = _deserialise_value(state)
        self.appliance_order = list(restored["appliance_order"])
        self.dictionary_components = dict(restored["dictionary_components"])
        self.reconstruction_bases = restored["reconstruction_bases"]
        self.disaggregation_bases = restored["disaggregation_bases"]
        self.component_slices = {
            key: tuple(int(value) for value in values)
            for key, values in restored["component_slices"].items()
        }
