"""Official Torch DSC algorithm behind the repository plugin interface.

Official source: https://github.com/nilmtk/nilmtk-contrib/blob/14efd545e09d836e02159b23444350af92c6ee70/nilmtk_contrib/torch/_dsc.py
Local adaptation: canonical household partitions and unified checkpoint/evaluation API.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np
import torch

from model_pipeline.api import FitResult, InferenceContext, PredictionOutput, TrainingContext
from model_pipeline.model_registry import register_model
from model_pipeline.models.classical.afhmm import _deserialise_value, _serialise_value


@dataclass(frozen=True)
class SparseCodeResult:
    codes: torch.Tensor
    objective: float
    iterations: int
    converged: bool


@dataclass(frozen=True)
class _DSCParameters:
    reconstruction_dictionary: tuple[tuple[float, ...], ...]
    discriminative_dictionary: tuple[tuple[float, ...], ...]
    training_windows: int
    reconstruction_objective: float
    reconstruction_iterations: int
    reconstruction_converged: bool
    activation_error: float


def _sparse_objective(dictionary, observations, codes, coefficient):
    residual = observations - dictionary @ codes
    return 0.5 * residual.square().sum() + coefficient * codes.sum()


@torch.no_grad()
def nonnegative_sparse_code(
    dictionary: torch.Tensor,
    observations: torch.Tensor,
    *,
    sparsity_coefficient: float,
    max_iterations: int = 200,
    tolerance: float = 1e-6,
) -> SparseCodeResult:
    """Official monotone accelerated proximal-gradient non-negative LASSO."""
    spectral_norm = torch.linalg.matrix_norm(dictionary, ord=2)
    lipschitz = spectral_norm.square()
    if not bool(torch.isfinite(lipschitz)) or float(lipschitz) <= 0:
        raise ValueError("dictionary must have a positive finite spectral norm.")
    step = lipschitz.reciprocal()
    threshold = step * sparsity_coefficient
    codes = torch.zeros(
        (dictionary.shape[1], observations.shape[1]),
        dtype=dictionary.dtype,
        device=dictionary.device,
    )
    extrapolated = codes.clone()
    momentum = 1.0
    objective = _sparse_objective(dictionary, observations, codes, sparsity_coefficient)
    converged = False
    for iteration in range(1, max_iterations + 1):
        gradient = dictionary.T @ (dictionary @ extrapolated - observations)
        candidate = torch.clamp(extrapolated - step * gradient - threshold, min=0)
        candidate_objective = _sparse_objective(
            dictionary, observations, candidate, sparsity_coefficient
        )
        if candidate_objective > objective:
            momentum = 1.0
            gradient = dictionary.T @ (dictionary @ codes - observations)
            candidate = torch.clamp(codes - step * gradient - threshold, min=0)
            candidate_objective = _sparse_objective(
                dictionary, observations, candidate, sparsity_coefficient
            )
        if not bool(torch.isfinite(candidate_objective)):
            raise RuntimeError("Sparse coding produced non-finite values.")
        change = torch.linalg.vector_norm(candidate - codes)
        scale = 1.0 + torch.linalg.vector_norm(codes)
        next_momentum = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * momentum**2))
        extrapolated = candidate + (momentum - 1.0) / next_momentum * (candidate - codes)
        codes, objective, momentum = candidate, candidate_objective, next_momentum
        if float(change) <= tolerance * float(scale):
            converged = True
            break
    return SparseCodeResult(codes, float(objective), iteration, converged)


def _normalize_dictionary(dictionary, fallback):
    dictionary = torch.clamp(dictionary, min=0)
    norms = torch.linalg.vector_norm(dictionary, dim=0)
    missing = norms <= torch.finfo(dictionary.dtype).eps
    if bool(missing.any()):
        replacement = torch.clamp(fallback, min=0)
        replacement_norm = torch.linalg.vector_norm(replacement)
        if float(replacement_norm) <= torch.finfo(dictionary.dtype).eps:
            replacement = torch.ones_like(replacement)
            replacement_norm = torch.linalg.vector_norm(replacement)
        dictionary[:, missing] = (replacement / replacement_norm)[:, None]
        norms = torch.linalg.vector_norm(dictionary, dim=0)
    return dictionary / norms.clamp_min(torch.finfo(dictionary.dtype).eps)


def _learn_dictionary(
    observations,
    *,
    n_components,
    sparsity_coefficient,
    dictionary_iterations,
    sparse_code_iterations,
    tolerance,
):
    if not bool((observations > 0).any()):
        raise ValueError("Appliance training data must contain positive power.")
    indices = torch.arange(n_components, device=observations.device).remainder(
        observations.shape[1]
    )
    fallback = observations.mean(dim=1)
    dictionary = _normalize_dictionary(observations[:, indices].clone(), fallback)
    best_dictionary = dictionary.clone()
    best_result = nonnegative_sparse_code(
        dictionary,
        observations,
        sparsity_coefficient=sparsity_coefficient,
        max_iterations=sparse_code_iterations,
        tolerance=tolerance,
    )
    best_objective = best_result.objective
    for _ in range(dictionary_iterations):
        code_result = nonnegative_sparse_code(
            dictionary,
            observations,
            sparsity_coefficient=sparsity_coefficient,
            max_iterations=sparse_code_iterations,
            tolerance=tolerance,
        )
        codes = code_result.codes
        lipschitz = torch.linalg.matrix_norm(codes, ord=2).square()
        if float(lipschitz) <= torch.finfo(dictionary.dtype).eps:
            break
        candidate = _normalize_dictionary(
            dictionary + (observations - dictionary @ codes) @ codes.T / lipschitz,
            fallback,
        )
        candidate_result = nonnegative_sparse_code(
            candidate,
            observations,
            sparsity_coefficient=sparsity_coefficient,
            max_iterations=sparse_code_iterations,
            tolerance=tolerance,
        )
        if candidate_result.objective >= best_objective:
            break
        relative = (best_objective - candidate_result.objective) / max(
            1.0, abs(best_objective)
        )
        best_dictionary, best_result = candidate.clone(), candidate_result
        best_objective, dictionary = candidate_result.objective, candidate
        if relative <= tolerance:
            break
    return best_dictionary, best_result


def _dictionary_step(aggregate, dictionary, predicted, target, learning_rate, fallback):
    gradient = ((aggregate - dictionary @ predicted) @ predicted.T) - (
        (aggregate - dictionary @ target) @ target.T
    )
    if float(torch.linalg.vector_norm(gradient)) <= torch.finfo(dictionary.dtype).eps:
        return None
    return _normalize_dictionary(dictionary - learning_rate * gradient, fallback)


def _fit_discriminative_dictionary(
    aggregate,
    reconstruction_dictionary,
    target_codes,
    *,
    sparsity_coefficient,
    discriminative_iterations,
    sparse_code_iterations,
    tolerance,
    learning_rate,
):
    dictionary = reconstruction_dictionary.clone()
    fallback = dictionary.mean(dim=1)
    validation_windows = int(aggregate.shape[1] * 0.2)
    if validation_windows:
        train_x, val_x = aggregate[:, :-validation_windows], aggregate[:, -validation_windows:]
        train_y, val_y = target_codes[:, :-validation_windows], target_codes[:, -validation_windows:]
    else:
        train_x = val_x = aggregate
        train_y = val_y = target_codes
    predicted = nonnegative_sparse_code(
        dictionary, train_x, sparsity_coefficient=sparsity_coefficient,
        max_iterations=sparse_code_iterations, tolerance=tolerance
    )
    val_predicted = predicted if not validation_windows else nonnegative_sparse_code(
        dictionary, val_x, sparsity_coefficient=sparsity_coefficient,
        max_iterations=sparse_code_iterations, tolerance=tolerance
    )
    best_error = float(torch.mean(torch.abs(val_predicted.codes - val_y)))
    best_dictionary = dictionary.clone()
    for _ in range(discriminative_iterations):
        candidate = _dictionary_step(
            train_x, dictionary, predicted.codes, train_y, learning_rate, fallback
        )
        if candidate is None:
            break
        change = torch.linalg.vector_norm(candidate - dictionary)
        scale = 1.0 + torch.linalg.vector_norm(dictionary)
        dictionary = candidate
        predicted = nonnegative_sparse_code(
            dictionary, train_x, sparsity_coefficient=sparsity_coefficient,
            max_iterations=sparse_code_iterations, tolerance=tolerance
        )
        val_predicted = predicted if not validation_windows else nonnegative_sparse_code(
            dictionary, val_x, sparsity_coefficient=sparsity_coefficient,
            max_iterations=sparse_code_iterations, tolerance=tolerance
        )
        error = float(torch.mean(torch.abs(val_predicted.codes - val_y)))
        if error < best_error:
            best_error, best_dictionary = error, dictionary.clone()
        if float(change) <= tolerance * float(scale):
            break
    return best_dictionary, best_error


def _windows(values, shape):
    values = torch.as_tensor(values, dtype=torch.float32).flatten()
    padding = (-values.numel()) % shape
    if padding:
        values = torch.nn.functional.pad(values, (0, padding))
    return values.reshape(-1, shape).T


def _matrix_tuple(values):
    return tuple(tuple(float(value) for value in row) for row in values.cpu().double())


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
        n_components: int = 10,
        sparsity_coef: float = 20.0,
        dictionary_iterations: int = 20,
        iterations: int = 20,
        sparse_code_iterations: int = 100,
        tolerance: float = 1e-5,
        learning_rate: float = 1e-9,
        enforce_aggregate: bool = True,
    ):
        if min(window_size, n_components, sparse_code_iterations) < 1:
            raise ValueError("DSC integer hyperparameters must be positive.")
        self.window_size = window_size
        self.n_components = n_components
        self.sparsity_coef = float(sparsity_coef)
        self.dictionary_iterations = dictionary_iterations
        self.iterations = iterations
        self.sparse_code_iterations = sparse_code_iterations
        self.tolerance = tolerance
        self.learning_rate = learning_rate
        self.enforce_aggregate = enforce_aggregate
        self._config = dict(
            window_size=window_size, n_components=n_components,
            sparsity_coef=sparsity_coef, dictionary_iterations=dictionary_iterations,
            iterations=iterations, sparse_code_iterations=sparse_code_iterations,
            tolerance=tolerance, learning_rate=learning_rate,
            enforce_aggregate=enforce_aggregate,
        )
        self.models: OrderedDict[str, _DSCParameters] = OrderedDict()

    def fit(self, train_data, validation_data, context: TrainingContext) -> FitResult:
        del validation_data
        mains = torch.cat([_windows(s.aggregate, self.window_size) for s in train_data.series], dim=1)
        names = tuple(train_data.series[0].appliances)
        targets = OrderedDict(
            (name, torch.cat([
                _windows(s.appliance_power[:, s.appliances.index(name)], self.window_size)
                for s in train_data.series
            ], dim=1)) for name in names
        )
        reconstruction, target_codes, results = OrderedDict(), OrderedDict(), OrderedDict()
        for name, observations in targets.items():
            dictionary, result = _learn_dictionary(
                observations, n_components=self.n_components,
                sparsity_coefficient=self.sparsity_coef,
                dictionary_iterations=self.dictionary_iterations,
                sparse_code_iterations=self.sparse_code_iterations,
                tolerance=self.tolerance,
            )
            reconstruction[name], target_codes[name], results[name] = dictionary, result.codes, result
        joint_reconstruction = torch.cat(tuple(reconstruction.values()), dim=1)
        joint_codes = torch.cat(tuple(target_codes.values()), dim=0)
        joint_discriminative, activation_error = _fit_discriminative_dictionary(
            mains, joint_reconstruction, joint_codes,
            sparsity_coefficient=self.sparsity_coef,
            discriminative_iterations=self.iterations,
            sparse_code_iterations=self.sparse_code_iterations,
            tolerance=self.tolerance, learning_rate=self.learning_rate,
        )
        fitted, start = OrderedDict(), 0
        for name, dictionary in reconstruction.items():
            stop = start + self.n_components
            fitted[name] = _DSCParameters(
                _matrix_tuple(dictionary), _matrix_tuple(joint_discriminative[:, start:stop]),
                int(mains.shape[1]), results[name].objective, results[name].iterations,
                results[name].converged, activation_error,
            )
            start = stop
        self.models = fitted
        validation = context.validate_candidate(epoch=1, model=self)
        return FitResult(
            history=[{"epoch": 1, "validation_mse": validation.mse}],
            best_epoch=1 if validation.improved else None,
            best_validation_mse=validation.mse,
            checkpoint_path=validation.checkpoint_path,
        )

    @torch.no_grad()
    def predict(self, inference_data, context: InferenceContext) -> PredictionOutput:
        del context
        if not self.models:
            raise RuntimeError("DSC must be fitted before prediction.")
        reconstruction = torch.cat([
            torch.tensor(m.reconstruction_dictionary, dtype=torch.float32)
            for m in self.models.values()
        ], dim=1)
        discriminative = torch.cat([
            torch.tensor(m.discriminative_dictionary, dtype=torch.float32)
            for m in self.models.values()
        ], dim=1)
        timestamps, households, powers = [], [], []
        for series in inference_data.series:
            windows = _windows(series.aggregate, self.window_size)
            codes = nonnegative_sparse_code(
                discriminative, windows, sparsity_coefficient=self.sparsity_coef,
                max_iterations=self.sparse_code_iterations, tolerance=self.tolerance,
            ).codes
            outputs, start = [], 0
            for _name in self.models:
                stop = start + self.n_components
                outputs.append((reconstruction[:, start:stop] @ codes[start:stop]).T.flatten()[:len(series.aggregate)])
                start = stop
            stacked = torch.stack(outputs).clamp_(min=0)
            aggregate = torch.as_tensor(series.aggregate, dtype=torch.float32)
            if self.enforce_aggregate:
                total = stacked.sum(dim=0)
                scale = torch.where(total > aggregate, aggregate / total.clamp_min(torch.finfo(total.dtype).eps), torch.ones_like(total))
                stacked *= scale
            timestamps.append(series.timestamps)
            households.append(np.repeat(series.household_id, len(series.timestamps)))
            powers.append(stacked.T.numpy())
        return PredictionOutput(
            timestamps=np.concatenate(timestamps), power=np.concatenate(powers),
            appliances=tuple(self.models), household_ids=np.concatenate(households),
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
        self.models = OrderedDict(
            (name, _DSCParameters(
                tuple(tuple(float(x) for x in row) for row in payload["reconstruction_dictionary"]),
                tuple(tuple(float(x) for x in row) for row in payload["discriminative_dictionary"]),
                int(payload["training_windows"]), float(payload["reconstruction_objective"]),
                int(payload["reconstruction_iterations"]), bool(payload["reconstruction_converged"]),
                float(payload["activation_error"]),
            )) for name, payload in restored.items()
        )
