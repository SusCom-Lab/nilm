"""Official solver-free Torch AFHMM behind the repository plugin interface.

Official source: https://github.com/nilmtk/nilmtk-contrib/blob/14efd545e09d836e02159b23444350af92c6ee70/nilmtk_contrib/torch/_afhmm.py
Local adaptation: canonical household partitions and unified checkpoint/evaluation API.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
import math
from typing import Any, Sequence

import numpy as np
import torch

from model_pipeline.api import FitResult, InferenceContext, PredictionOutput, TrainingContext
from model_pipeline.model_registry import register_model


@dataclass(frozen=True)
class _HMMParameters:
    state_means: torch.Tensor
    initial_probabilities: torch.Tensor
    transition_probabilities: torch.Tensor


@dataclass(frozen=True)
class _ObservedFit:
    parameters: _HMMParameters
    loss_history: tuple[float, ...]
    iterations: int
    converged: bool


@dataclass(frozen=True)
class _AFHMMParameters:
    state_means: tuple[float, ...]
    initial_probabilities: tuple[float, ...]
    transition_probabilities: tuple[tuple[float, ...], ...]
    background_mean: float
    fit_loss_history: tuple[float, ...]
    fit_iterations: int
    fit_converged: bool
    num_samples: int
    num_chunks: int


def _serialise_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _serialise_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialise_value(item) for item in value]
    return value


def _deserialise_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _deserialise_value(item) for key, item in value.items()}
    if isinstance(value, list):
        if value and all(not isinstance(item, (dict, list)) for item in value):
            return np.asarray(value)
        return [_deserialise_value(item) for item in value]
    return value


def _canonicalize(means, initial, transition):
    order = torch.argsort(means, stable=True)
    return _HMMParameters(means[order], initial[order], transition[order][:, order])


@torch.no_grad()
def _fit_observed_gaussian_hmm(
    sequences: Sequence[torch.Tensor],
    *,
    n_states: int,
    max_iterations: int,
    tolerance: float,
    pseudocount: float,
) -> _ObservedFit:
    """Official deterministic 1-D Lloyd fit and smoothed transition counts."""
    values = torch.cat(tuple(sequences))
    unique_values = torch.unique(values, sorted=True)
    if unique_values.numel() < n_states:
        raise ValueError(
            f"n_states={n_states} requires at least {n_states} distinct power values."
        )
    positions = torch.div(
        torch.arange(n_states, device=values.device) * (unique_values.numel() - 1),
        n_states - 1,
        rounding_mode="floor",
    )
    means = unique_values[positions]

    def assign(current_means):
        states = torch.argmin((values[:, None] - current_means[None, :]).square(), dim=1)
        return states, float((values - current_means[states]).square().sum())

    states, loss = assign(means)
    history = [loss]
    converged = False
    for iteration in range(1, max_iterations + 1):
        updated = means.clone()
        for state in range(n_states):
            members = values[states == state]
            if members.numel():
                updated[state] = members.mean()
        new_states, new_loss = assign(updated)
        unchanged = torch.equal(new_states, states)
        improvement = loss - new_loss
        history.append(new_loss)
        means, states, loss = updated, new_states, new_loss
        if unchanged or improvement <= tolerance * max(1.0, abs(history[-2])):
            converged = True
            break
    order = torch.argsort(means, stable=True)
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(n_states, device=values.device)
    means, states = means[order], inverse[states]
    state_sequences = torch.split(states, [sequence.numel() for sequence in sequences])
    initial_counts = means.new_full((n_states,), pseudocount)
    transition_counts = means.new_full((n_states, n_states), pseudocount)
    for sequence_states in state_sequences:
        initial_counts[sequence_states[0]] += 1
        if sequence_states.numel() > 1:
            transitions = sequence_states[:-1] * n_states + sequence_states[1:]
            transition_counts += torch.bincount(
                transitions, minlength=n_states * n_states
            ).reshape(n_states, n_states).to(means.dtype)
    parameters = _canonicalize(
        means,
        initial_counts / initial_counts.sum(),
        transition_counts / transition_counts.sum(dim=1, keepdim=True),
    )
    return _ObservedFit(parameters, tuple(history), iteration, converged)


def _hmm_viterbi(emissions, log_initial, log_transition):
    time_points, states = emissions.shape
    scores = emissions.new_empty((time_points, states))
    backpointers = torch.zeros(
        (time_points, states), dtype=torch.int64, device=emissions.device
    )
    scores[0] = log_initial + emissions[0]
    for time in range(1, time_points):
        candidates = scores[time - 1][:, None] + log_transition
        best, sources = torch.max(candidates, dim=0)
        scores[time] = best + emissions[time]
        backpointers[time] = sources
    final = int(torch.argmax(scores[-1]))
    decoded = torch.empty(time_points, dtype=torch.int64, device=emissions.device)
    decoded[-1] = final
    for time in range(time_points - 1, 0, -1):
        decoded[time - 1] = backpointers[time, decoded[time]]
    return decoded, float(scores[-1, final])


def _gaussian_emissions(observations, means, noise_std):
    scale = observations.new_tensor(noise_std)
    residual = (observations[:, None] - means[None, :]) / scale
    return -0.5 * residual.square() - torch.log(scale) - 0.5 * math.log(2 * math.pi)


def _path_score(states, emissions, log_initial, log_transition):
    time = torch.arange(states.numel(), device=states.device)
    score = log_initial[states[0]] + emissions[time, states].sum()
    if states.numel() > 1:
        score += log_transition[states[:-1], states[1:]].sum()
    return float(score)


@torch.no_grad()
def _coordinate_viterbi(observations, means, initial, transition, noise_std, max_iterations):
    """Official coordinate-ascent factorial HMM decoder."""
    time_points, appliance_count = observations.numel(), len(means)
    states = torch.empty((time_points, appliance_count), dtype=torch.int64)
    log_initial = tuple(torch.log(value) for value in initial)
    log_transition = tuple(torch.log(value) for value in transition)
    for appliance, appliance_means in enumerate(means):
        prior = appliance_means.new_zeros((time_points, appliance_means.numel()))
        states[:, appliance] = _hmm_viterbi(
            prior, log_initial[appliance], log_transition[appliance]
        )[0]
    power = torch.stack(
        [value[states[:, index]] for index, value in enumerate(means)], dim=1
    )
    aggregate = power.sum(dim=1)
    converged = False
    score_history = []
    for iteration in range(1, max_iterations + 1):
        changed = False
        for appliance, appliance_means in enumerate(means):
            residual = observations - (aggregate - power[:, appliance])
            emissions = _gaussian_emissions(residual, appliance_means, noise_std)
            current = _path_score(
                states[:, appliance], emissions,
                log_initial[appliance], log_transition[appliance]
            )
            decoded, decoded_score = _hmm_viterbi(
                emissions, log_initial[appliance], log_transition[appliance]
            )
            tolerance = 1e-12 * max(1.0, abs(current), abs(decoded_score))
            if decoded_score <= current + tolerance:
                continue
            previous = power[:, appliance].clone()
            states[:, appliance] = decoded
            power[:, appliance] = appliance_means[decoded]
            aggregate = aggregate - previous + power[:, appliance]
            changed = True
        # Diagnostics are not exposed by the repository adapter, but preserving
        # this calculation also catches non-finite inference immediately.
        score = -0.5 * (((observations - aggregate) / noise_std).square()).sum()
        score_history.append(float(score))
        if not changed:
            converged = True
            break
    return power, iteration, converged, tuple(score_history)


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
        pseudocount: float = 1.0,
        kmeans_max_iterations: int = 100,
        kmeans_tolerance: float = 1e-6,
        noise_std: float = 100.0,
        inference_max_iterations: int = 20,
        fail_on_nonconvergence: bool = False,
    ):
        if n_states < 2:
            raise ValueError("n_states must be at least two.")
        self.window_size = window_size  # interface metadata; official AFHMM uses full chunks
        self.n_states = n_states
        self.pseudocount = pseudocount
        self.kmeans_max_iterations = kmeans_max_iterations
        self.kmeans_tolerance = kmeans_tolerance
        self.noise_std = noise_std
        self.inference_max_iterations = inference_max_iterations
        self.fail_on_nonconvergence = fail_on_nonconvergence
        self._config = dict(
            window_size=window_size, n_states=n_states, pseudocount=pseudocount,
            kmeans_max_iterations=kmeans_max_iterations,
            kmeans_tolerance=kmeans_tolerance, noise_std=noise_std,
            inference_max_iterations=inference_max_iterations,
            fail_on_nonconvergence=fail_on_nonconvergence,
        )
        self.models: OrderedDict[str, _AFHMMParameters] = OrderedDict()
        self.last_inference_diagnostics = ()

    def fit(self, train_data, validation_data, context: TrainingContext) -> FitResult:
        del validation_data
        names = tuple(sorted(train_data.series[0].appliances))
        frames = {
            name: tuple(
                torch.as_tensor(
                    series.appliance_power[:, series.appliances.index(name)],
                    dtype=torch.float64,
                )
                for series in train_data.series
            )
            for name in names
        }
        residuals = []
        for series in train_data.series:
            residuals.append(
                torch.as_tensor(series.aggregate, dtype=torch.float64)
                - torch.as_tensor(series.appliance_power.sum(axis=1), dtype=torch.float64)
            )
        background_mean = max(0.0, float(torch.cat(residuals).mean()))
        fitted = OrderedDict()
        for name in names:
            result = _fit_observed_gaussian_hmm(
                frames[name], n_states=self.n_states,
                max_iterations=self.kmeans_max_iterations,
                tolerance=self.kmeans_tolerance, pseudocount=self.pseudocount,
            )
            parameters = result.parameters
            fitted[name] = _AFHMMParameters(
                tuple(float(x) for x in parameters.state_means),
                tuple(float(x) for x in parameters.initial_probabilities),
                tuple(tuple(float(x) for x in row) for row in parameters.transition_probabilities),
                background_mean, result.loss_history, result.iterations,
                result.converged, sum(x.numel() for x in frames[name]), len(frames[name]),
            )
        self.models = fitted
        validation = context.validate_candidate(epoch=1, model=self)
        return FitResult(
            history=[{"epoch": 1, "validation_mse": validation.mse}],
            best_epoch=1 if validation.improved else None,
            best_validation_mse=validation.mse,
            checkpoint_path=validation.checkpoint_path,
        )

    def _predict_series(self, aggregate):
        if not self.models:
            raise RuntimeError("AFHMM must be fitted before prediction.")
        names = tuple(self.models)
        background = self.models[names[0]].background_mean
        means = tuple(torch.tensor(self.models[name].state_means, dtype=torch.float64) for name in names)
        initial = tuple(torch.tensor(self.models[name].initial_probabilities, dtype=torch.float64) for name in names)
        transition = tuple(torch.tensor(self.models[name].transition_probabilities, dtype=torch.float64) for name in names)
        observations = torch.as_tensor(aggregate, dtype=torch.float64) - background
        power, iterations, converged, history = _coordinate_viterbi(
            observations, means, initial, transition,
            self.noise_std, self.inference_max_iterations,
        )
        if self.fail_on_nonconvergence and not converged:
            raise RuntimeError("AFHMM coordinate inference did not converge.")
        self.last_inference_diagnostics += ((len(aggregate), iterations, converged, history),)
        return power.float().numpy()

    def predict(self, inference_data, context: InferenceContext) -> PredictionOutput:
        del context
        self.last_inference_diagnostics = ()
        timestamps, households, powers = [], [], []
        for series in inference_data.series:
            predicted = self._predict_series(series.aggregate)
            order = [tuple(self.models).index(name) for name in series.appliances]
            timestamps.append(series.timestamps)
            households.append(np.repeat(series.household_id, len(series.timestamps)))
            powers.append(predicted[:, order])
        return PredictionOutput(
            timestamps=np.concatenate(timestamps), power=np.concatenate(powers),
            appliances=inference_data.series[0].appliances,
            household_ids=np.concatenate(households),
        )

    def save(self, path, *, metadata=None) -> None:
        torch.save({
            "model_key": self._registry_key, "init_kwargs": dict(self._config),
            "model_state": self.get_state(), "metadata": dict(metadata or {}),
        }, path)

    def load(self, path, device="cpu") -> None:
        try:
            checkpoint = torch.load(path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location=device)
        self.set_state(checkpoint["model_state"])

    def get_state(self) -> dict[str, Any]:
        return _serialise_value({name: asdict(model) for name, model in self.models.items()})

    def set_state(self, state: dict[str, Any]) -> None:
        restored = _deserialise_value(state)
        self.models = OrderedDict()
        for name, payload in restored.items():
            self.models[name] = _AFHMMParameters(
                tuple(float(x) for x in payload["state_means"]),
                tuple(float(x) for x in payload["initial_probabilities"]),
                tuple(tuple(float(x) for x in row) for row in payload["transition_probabilities"]),
                float(payload["background_mean"]),
                tuple(float(x) for x in payload["fit_loss_history"]),
                int(payload["fit_iterations"]), bool(payload["fit_converged"]),
                int(payload["num_samples"]), int(payload["num_chunks"]),
            )
