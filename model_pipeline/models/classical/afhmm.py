"""
AFHMM baselines adapted to joint multi-appliance disaggregation.

Official source: https://github.com/nilmtk/nilmtk-contrib/blob/14efd545e09d836e02159b23444350af92c6ee70/nilmtk_contrib/torch/afhmm.py
Local adaptation: joint-house repository plugin interface.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from dataclasses import dataclass

import torch

from model_pipeline.api import (
    FitResult,
    InferenceContext,
    PredictionOutput,
    TrainingContext,
)
from model_pipeline.model_registry import register_model


@dataclass(frozen=True)
class ClassicalGroup:
    group_id: str
    time: np.ndarray
    aggregate: np.ndarray
    appliances: dict[str, np.ndarray]


@dataclass(frozen=True)
class ClassicalInferenceGroup:
    group_id: str
    time: np.ndarray
    aggregate: np.ndarray


def supervised_groups(partition) -> dict[str, ClassicalGroup]:
    return {
        series.household_id: ClassicalGroup(
            group_id=series.household_id,
            time=series.timestamps,
            aggregate=series.aggregate,
            appliances={
                name: series.appliance_power[:, index]
                for index, name in enumerate(series.appliances)
            },
        )
        for series in partition.series
    }


def inference_groups(partition) -> dict[str, ClassicalInferenceGroup]:
    return {
        series.household_id: ClassicalInferenceGroup(
            group_id=series.household_id,
            time=series.timestamps,
            aggregate=series.aggregate,
        )
        for series in partition.series
    }


def _fit_gaussian_hmm(values: np.ndarray, n_states: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        from hmmlearn import hmm
    except ImportError as exc:
        raise ImportError("AFHMM requires 'hmmlearn'. Install it with `pip install hmmlearn`.") from exc

    values = np.asarray(values, dtype=np.float32).reshape(-1, 1)
    if values.size == 0:
        raise ValueError("Cannot fit AFHMM on empty values.")

    model = hmm.GaussianHMM(n_components=n_states, covariance_type="full")
    model.fit(values)
    means = model.means_.flatten().reshape(-1, 1).astype(np.float32)
    states = model.predict(values)
    transmat = np.clip(model.transmat_.T.astype(np.float32), 1e-8, None)
    counts = Counter(states.flatten())
    priors = np.zeros(n_states, dtype=np.float32)
    total = max(1, sum(counts.values()))
    for idx in range(n_states):
        priors[idx] = counts.get(idx, 0) / total
    priors = np.clip(priors, 1e-8, None)
    return means, priors, transmat


def _serialise_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _serialise_value(subvalue) for key, subvalue in value.items()}
    if isinstance(value, list):
        return [_serialise_value(item) for item in value]
    return value


def _deserialise_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _deserialise_value(subvalue) for key, subvalue in value.items()}
    if isinstance(value, list):
        if value and all(not isinstance(item, (dict, list)) for item in value):
            return np.asarray(value, dtype=np.float32)
        return [_deserialise_value(item) for item in value]
    return value


@register_model("afhmm", aliases=("AFHMM",), display_name="AFHMM")
class AFHMMBaseline:
    display_name = "AFHMM"
    model_family = "probabilistic"
    target_type = "sequence"
    supports_gradient = False
    is_joint_classical = True
    default_num_epochs = 1
    official_batch_size = 1

    def __init__(
        self,
        *,
        window_size: int = 599,
        n_states: int = 2,
        optimisation_epochs: int = 6,
        sigma_floor: float = 1.0,
        solver: str = "SCS",
    ):
        if (window_size, n_states, optimisation_epochs, sigma_floor, solver) != (
            599, 2, 6, 1.0, "SCS"
        ):
            raise ValueError("AFHMM uses the fixed repository reference defaults.")
        self.window_size = window_size
        self._config = {
            "window_size": window_size,
            "n_states": n_states,
            "optimisation_epochs": optimisation_epochs,
            "sigma_floor": sigma_floor,
            "solver": solver,
        }
        self.n_states = n_states
        self.optimisation_epochs = optimisation_epochs
        self.sigma_floor = sigma_floor
        self.solver = solver
        self.appliance_order: list[str] = []
        self.means_vector: dict[str, np.ndarray] = {}
        self.pi_vector: dict[str, np.ndarray] = {}
        self.transmat_vector: dict[str, np.ndarray] = {}
        self.signal_aggregates: dict[str, float] = {}

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

    def fit_joint(self, grouped_train_data: dict[str, ClassicalGroup]) -> None:
        if not grouped_train_data:
            raise ValueError("AFHMMBaseline requires non-empty grouped training data.")

        appliance_names = sorted(next(iter(grouped_train_data.values())).appliances.keys())
        self.appliance_order = appliance_names
        concatenated_targets: dict[str, list[np.ndarray]] = {name: [] for name in appliance_names}
        for group in grouped_train_data.values():
            if sorted(group.appliances.keys()) != appliance_names:
                raise ValueError("All grouped training entries must contain the same appliance set.")
            for appliance_name in appliance_names:
                concatenated_targets[appliance_name].append(group.appliances[appliance_name])

        self.means_vector = {}
        self.pi_vector = {}
        self.transmat_vector = {}
        self.signal_aggregates = {}
        for appliance_name in appliance_names:
            values = np.concatenate(concatenated_targets[appliance_name], axis=0)
            means, priors, transmat = _fit_gaussian_hmm(values, self.n_states)
            self.means_vector[appliance_name] = means
            self.pi_vector[appliance_name] = priors
            self.transmat_vector[appliance_name] = transmat
            self.signal_aggregates[appliance_name] = float(np.mean(values))

    def _build_joint_constraints(self, length: int):
        import cvxpy as cvx

        constraints: list[Any] = []
        state_vectors: dict[str, Any] = {}
        transition_matrices: dict[str, list[Any]] = {}
        for appliance_name in self.appliance_order:
            state_vector = cvx.Variable((length, self.n_states), name=f"{appliance_name}_state_vec")
            variable_matrices = [
                cvx.Variable((self.n_states, self.n_states), name=f"{appliance_name}_transition_{idx}")
                for idx in range(length)
            ]
            state_vectors[appliance_name] = state_vector
            transition_matrices[appliance_name] = variable_matrices
            constraints.extend([state_vector >= 0, state_vector <= 1])
            for t in range(length):
                constraints.append(cvx.sum(state_vector[t]) == 1)
                constraints.extend([variable_matrices[t] >= 0, variable_matrices[t] <= 1])
                for state_idx in range(self.n_states):
                    constraints.append(cvx.sum(variable_matrices[t].T[state_idx]) == state_vector[t][state_idx])
            for t in range(1, length):
                for state_idx in range(self.n_states):
                    constraints.append(cvx.sum(variable_matrices[t][state_idx]) == state_vector[t - 1][state_idx])
        return state_vectors, transition_matrices, constraints

    def _maybe_add_constraints(self, constraints: list[Any], state_vectors: dict[str, Any]) -> None:
        del constraints, state_vectors

    def _solve_group(self, aggregate_sequence: np.ndarray) -> pd.DataFrame:
        import cvxpy as cvx

        if not self.appliance_order:
            raise RuntimeError("AFHMMBaseline must be fitted before disaggregation.")

        aggregate_sequence = np.asarray(aggregate_sequence, dtype=np.float32).reshape(-1, 1)
        length = len(aggregate_sequence)
        sigma = 100 * np.ones((length, 1), dtype=np.float32)

        state_vectors, transition_matrices, constraints = self._build_joint_constraints(length)
        self._maybe_add_constraints(constraints, state_vectors)

        total_usage = np.zeros((length, 1), dtype=np.float32)
        for appliance_name in self.appliance_order:
            total_usage = total_usage + state_vectors[appliance_name] @ self.means_vector[appliance_name]

        term_1 = 0
        term_2 = 0
        for appliance_name in self.appliance_order:
            for matrix in transition_matrices[appliance_name]:
                term_1 -= cvx.sum(cvx.multiply(matrix, np.log(self.transmat_vector[appliance_name])))
            term_2 -= cvx.sum(cvx.multiply(state_vectors[appliance_name][0], np.log(self.pi_vector[appliance_name])))

        solved_states: dict[str, np.ndarray] | None = None
        for epoch in range(self.optimisation_epochs):
            if epoch % 2 == 1 and solved_states is not None:
                usage = np.zeros((length,), dtype=np.float32)
                for appliance_name in self.appliance_order:
                    usage += np.sum(solved_states[appliance_name] @ self.means_vector[appliance_name], axis=1)
                residual = (aggregate_sequence.flatten() - usage).reshape(-1, 1)
                sigma = np.where(np.abs(residual) < self.sigma_floor, self.sigma_floor, np.abs(residual))
                continue

            term_3 = 0
            term_4 = 0
            for idx in range(length):
                term_4 += 0.5 * ((aggregate_sequence[idx][0] - total_usage[idx][0]) ** 2 / (sigma[idx][0] ** 2))
                term_3 += 0.5 * np.log(sigma[idx][0] ** 2)

            objective = cvx.Minimize(term_1 + term_2 + term_3 + term_4)
            problem = cvx.Problem(objective, constraints)
            problem.solve(solver=self.solver, verbose=False, warm_start=True)
            solved_states = {
                appliance_name: state_vectors[appliance_name].value
                for appliance_name in self.appliance_order
            }

        if solved_states is None:
            raise RuntimeError("AFHMM optimisation failed to produce a state solution.")

        prediction_dict = {}
        for appliance_name in self.appliance_order:
            prediction_dict[appliance_name] = np.sum(
                solved_states[appliance_name] @ self.means_vector[appliance_name],
                axis=1,
            ).astype(np.float32)
        return pd.DataFrame(prediction_dict, dtype="float32")

    def disaggregate_joint(
        self,
        grouped_test_data: dict[str, ClassicalInferenceGroup],
    ) -> dict[str, pd.DataFrame]:
        predictions: dict[str, pd.DataFrame] = {}
        for group_id, group in grouped_test_data.items():
            predictions[group_id] = self._solve_group(group.aggregate)
        return predictions

    def predict(self, inference_data, context: InferenceContext) -> PredictionOutput:
        del context
        grouped = inference_groups(inference_data)
        predictions = self.disaggregate_joint(grouped)
        timestamps, households, powers = [], [], []
        for series in inference_data.series:
            frame = predictions[series.household_id]
            missing = [name for name in series.appliances if name not in frame]
            if missing:
                raise ValueError(f"AFHMM prediction is missing appliances: {missing}.")
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
            },
            path,
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
                "means_vector": self.means_vector,
                "pi_vector": self.pi_vector,
                "transmat_vector": self.transmat_vector,
                "signal_aggregates": self.signal_aggregates,
            }
        )

    def set_state(self, state: dict[str, Any]) -> None:
        restored = _deserialise_value(state)
        self.appliance_order = list(restored["appliance_order"])
        self.means_vector = dict(restored["means_vector"])
        self.pi_vector = dict(restored["pi_vector"])
        self.transmat_vector = dict(restored["transmat_vector"])
        self.signal_aggregates = {
            key: float(value) for key, value in restored.get("signal_aggregates", {}).items()
        }
