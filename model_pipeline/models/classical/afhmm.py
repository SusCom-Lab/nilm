"""
Classical AFHMM baselines for NILM.

This file contains the AFHMM baseline and the AFHMM_SAC variant. These
implementations are implemented for this project based on the cited papers. A
confirmed related open-source baseline collection covering these algorithms is:
https://github.com/nilmtk/nilmtk-contrib

Paper references:
    Zhong, M., Goddard, N., and Sutton, C. (2014).
    "Signal Aggregate Constraints in Additive Factorial HMMs, with Application
    to Energy Disaggregation." Advances in Neural Information Processing
    Systems 27.

    Kolter, J. Z., and Jaakkola, T. (2012).
    "Approximate Inference in Additive Factorial HMMs with Application to
    Energy Disaggregation." Proceedings of the 15th International Conference
    on Artificial Intelligence and Statistics.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from model_pipeline.model_registry import register_model
from model_pipeline.models.base_model import ClassicalNILMModel


def _ensure_2d(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 1:
        return array[:, None]
    return array


class _WindowClassicalModel(ClassicalNILMModel):
    target_type = "sequence"
    default_output_size = None

    def __init__(self, *, window_size: int = 599, **kwargs):
        super().__init__(window_size=window_size, **kwargs)

    def _serialise_arrays(self, state: dict[str, Any]) -> dict[str, Any]:
        serialised = {}
        for key, value in state.items():
            serialised[key] = value.tolist() if isinstance(value, np.ndarray) else value
        return serialised

    def _deserialise_arrays(self, state: dict[str, Any]) -> dict[str, Any]:
        restored = {}
        for key, value in state.items():
            restored[key] = np.asarray(value, dtype=np.float32) if isinstance(value, list) else value
        return restored


@register_model("afhmm", aliases=("AFHMM",), display_name="AFHMM")
class AFHMMBaseline(_WindowClassicalModel):
    display_name = "AFHMM"
    model_family = "probabilistic"

    def __init__(
        self,
        *,
        window_size: int = 599,
        n_appliance_states: int = 3,
        n_background_states: int = 3,
        max_training_windows: int = 1024,
        transition_smoothing: float = 1e-3,
        **kwargs,
    ):
        super().__init__(
            window_size=window_size,
            n_appliance_states=n_appliance_states,
            n_background_states=n_background_states,
            max_training_windows=max_training_windows,
            transition_smoothing=transition_smoothing,
            **kwargs,
        )
        self.n_appliance_states = n_appliance_states
        self.n_background_states = n_background_states
        self.max_training_windows = max_training_windows
        self.transition_smoothing = transition_smoothing
        self.appliance_means = None
        self.background_means = None
        self.appliance_log_transitions = None
        self.background_log_transitions = None
        self.appliance_log_prior = None
        self.background_log_prior = None

    def _fit_chain(self, values: np.ndarray, n_states: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        if values.size == 0:
            raise ValueError("Cannot fit AFHMM on empty values.")

        quantiles = np.linspace(0.0, 1.0, num=n_states + 2)[1:-1]
        means = np.quantile(values, quantiles) if n_states > 1 else np.array([values.mean()], dtype=np.float32)
        means = np.sort(np.asarray(means, dtype=np.float32))

        for _ in range(8):
            distances = np.abs(values[:, None] - means[None, :])
            assignments = np.argmin(distances, axis=1)
            updated = means.copy()
            for idx in range(n_states):
                cluster = values[assignments == idx]
                if cluster.size > 0:
                    updated[idx] = cluster.mean()
            if np.allclose(updated, means):
                break
            means = np.sort(updated)

        distances = np.abs(values[:, None] - means[None, :])
        assignments = np.argmin(distances, axis=1)
        transition_counts = np.full((n_states, n_states), self.transition_smoothing, dtype=np.float32)
        for prev_state, next_state in zip(assignments[:-1], assignments[1:]):
            transition_counts[prev_state, next_state] += 1.0

        transitions = transition_counts / transition_counts.sum(axis=1, keepdims=True)
        priors = np.bincount(assignments, minlength=n_states).astype(np.float32) + self.transition_smoothing
        priors /= priors.sum()
        return means, np.log(transitions), np.log(priors)

    def fit(self, aggregate_windows: np.ndarray, target_windows: np.ndarray) -> None:
        aggregate_windows = _ensure_2d(aggregate_windows)
        target_windows = _ensure_2d(target_windows)
        sample_size = min(len(aggregate_windows), self.max_training_windows)
        sample_indices = np.linspace(0, len(aggregate_windows) - 1, num=sample_size, dtype=int)
        aggregate_sample = aggregate_windows[sample_indices]
        target_sample = target_windows[sample_indices]
        background_sample = aggregate_sample - target_sample

        self.appliance_means, self.appliance_log_transitions, self.appliance_log_prior = self._fit_chain(
            target_sample.reshape(-1),
            self.n_appliance_states,
        )
        self.background_means, self.background_log_transitions, self.background_log_prior = self._fit_chain(
            background_sample.reshape(-1),
            self.n_background_states,
        )

    def _viterbi_disaggregate(self, aggregate_sequence: np.ndarray) -> np.ndarray:
        if self.appliance_means is None or self.background_means is None:
            raise RuntimeError("AFHMMBaseline must be fitted before disaggregation.")

        aggregate_sequence = np.asarray(aggregate_sequence, dtype=np.float32).reshape(-1)
        n_steps = aggregate_sequence.size
        n_joint_states = self.n_appliance_states * self.n_background_states
        scores = np.full((n_steps, n_joint_states), -np.inf, dtype=np.float32)
        backpointers = np.zeros((n_steps, n_joint_states), dtype=np.int32)
        emission_means = np.array(
            [
                self.appliance_means[a_idx] + self.background_means[b_idx]
                for a_idx in range(self.n_appliance_states)
                for b_idx in range(self.n_background_states)
            ],
            dtype=np.float32,
        )

        for joint_idx in range(n_joint_states):
            a_idx = joint_idx // self.n_background_states
            b_idx = joint_idx % self.n_background_states
            prior = self.appliance_log_prior[a_idx] + self.background_log_prior[b_idx]
            emission = -((aggregate_sequence[0] - emission_means[joint_idx]) ** 2)
            scores[0, joint_idx] = prior + emission

        for time_idx in range(1, n_steps):
            for joint_idx in range(n_joint_states):
                a_idx = joint_idx // self.n_background_states
                b_idx = joint_idx % self.n_background_states
                emission = -((aggregate_sequence[time_idx] - emission_means[joint_idx]) ** 2)
                transition_scores = np.empty(n_joint_states, dtype=np.float32)
                for prev_joint_idx in range(n_joint_states):
                    prev_a_idx = prev_joint_idx // self.n_background_states
                    prev_b_idx = prev_joint_idx % self.n_background_states
                    transition_scores[prev_joint_idx] = (
                        scores[time_idx - 1, prev_joint_idx]
                        + self.appliance_log_transitions[prev_a_idx, a_idx]
                        + self.background_log_transitions[prev_b_idx, b_idx]
                    )
                best_prev = int(np.argmax(transition_scores))
                scores[time_idx, joint_idx] = transition_scores[best_prev] + emission
                backpointers[time_idx, joint_idx] = best_prev

        best_last = int(np.argmax(scores[-1]))
        state_path = [best_last]
        for time_idx in range(n_steps - 1, 0, -1):
            state_path.append(int(backpointers[time_idx, state_path[-1]]))
        state_path.reverse()

        prediction = np.zeros(n_steps, dtype=np.float32)
        for time_idx, joint_idx in enumerate(state_path):
            appliance_state = joint_idx // self.n_background_states
            prediction[time_idx] = self.appliance_means[appliance_state]
        return prediction

    def disaggregate(self, inputs: np.ndarray) -> np.ndarray:
        inputs = _ensure_2d(inputs)
        return np.vstack([self._viterbi_disaggregate(window) for window in inputs])

    def get_state(self) -> dict[str, Any]:
        return self._serialise_arrays(
            {
                "appliance_means": self.appliance_means,
                "background_means": self.background_means,
                "appliance_log_transitions": self.appliance_log_transitions,
                "background_log_transitions": self.background_log_transitions,
                "appliance_log_prior": self.appliance_log_prior,
                "background_log_prior": self.background_log_prior,
            }
        )

    def set_state(self, state: dict[str, Any]) -> None:
        state = self._deserialise_arrays(state)
        self.appliance_means = state["appliance_means"]
        self.background_means = state["background_means"]
        self.appliance_log_transitions = state["appliance_log_transitions"]
        self.background_log_transitions = state["background_log_transitions"]
        self.appliance_log_prior = state["appliance_log_prior"]
        self.background_log_prior = state["background_log_prior"]


@register_model("afhmm_sac", aliases=("AFHMM-SAC",), display_name="AFHMM_SAC")
class AFHMMSACBaseline(AFHMMBaseline):
    display_name = "AFHMM_SAC"

    def __init__(self, *, window_size: int = 599, smoothness_weight: float = 0.05, **kwargs):
        super().__init__(window_size=window_size, smoothness_weight=smoothness_weight, **kwargs)
        self.smoothness_weight = smoothness_weight

    def _project_signal(self, base_prediction: np.ndarray, aggregate_sequence: np.ndarray) -> np.ndarray:
        aggregate_sequence = np.asarray(aggregate_sequence, dtype=np.float32).reshape(-1)
        base_prediction = np.asarray(base_prediction, dtype=np.float32).reshape(-1)

        try:
            import cvxpy as cp

            variable = cp.Variable(base_prediction.size)
            objective = cp.Minimize(
                cp.sum_squares(variable - base_prediction)
                + self.smoothness_weight * cp.sum_squares(variable[1:] - variable[:-1])
            )
            constraints = [variable >= 0, variable <= aggregate_sequence]
            problem = cp.Problem(objective, constraints)
            problem.solve(solver=cp.OSQP, warm_start=True, verbose=False)
            if variable.value is not None:
                return np.asarray(variable.value, dtype=np.float32)
        except Exception:
            pass

        clipped = np.clip(base_prediction, 0.0, aggregate_sequence)
        if clipped.size < 3:
            return clipped
        kernel = np.array([0.25, 0.5, 0.25], dtype=np.float32)
        padded = np.pad(clipped, (1, 1), mode="edge")
        smoothed = np.convolve(padded, kernel, mode="valid")
        return np.clip(smoothed, 0.0, aggregate_sequence)

    def disaggregate(self, inputs: np.ndarray) -> np.ndarray:
        base_predictions = super().disaggregate(inputs)
        refined = [
            self._project_signal(base_prediction, aggregate_window)
            for base_prediction, aggregate_window in zip(base_predictions, _ensure_2d(inputs))
        ]
        return np.vstack(refined)
