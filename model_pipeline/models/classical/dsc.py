"""
Discriminative sparse coding baseline for NILM.

This DSC implementation is implemented for this project based on the cited
paper. A confirmed related open-source baseline collection covering DSC is:
https://github.com/nilmtk/nilmtk-contrib

Paper references:
    Kolter, J. Z., Batra, S., and Ng, A. Y. (2010).
    "Energy Disaggregation via Discriminative Sparse Coding."
    Advances in Neural Information Processing Systems 23.

    Batra, N., Kukunuri, R., Pandey, A., Malakar, R., Kumar, R.,
    Krystalakos, O., Zhong, M., Meira, P., and Parson, O. (2019).
    "Towards Reproducible State-of-the-Art Energy Disaggregation."
    Proceedings of the 6th ACM International Conference on Systems for
    Energy-Efficient Buildings, Cities, and Transportation.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from model_pipeline.model_registry import register_model
from model_pipeline.models.classical.afhmm import _WindowClassicalModel, _ensure_2d


def _soft_threshold(x: np.ndarray, threshold: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)


@register_model("dsc", aliases=("DSC",), display_name="DSC")
class DiscriminativeSparseCoding(_WindowClassicalModel):
    display_name = "DSC"
    model_family = "probabilistic"

    def __init__(
        self,
        *,
        window_size: int = 599,
        n_atoms: int = 16,
        sparse_lambda: float = 0.1,
        ista_steps: int = 25,
        dictionary_steps: int = 6,
        max_training_windows: int = 1024,
        ridge_alpha: float = 1e-3,
        **kwargs,
    ):
        super().__init__(
            window_size=window_size,
            n_atoms=n_atoms,
            sparse_lambda=sparse_lambda,
            ista_steps=ista_steps,
            dictionary_steps=dictionary_steps,
            max_training_windows=max_training_windows,
            ridge_alpha=ridge_alpha,
            **kwargs,
        )
        self.n_atoms = n_atoms
        self.sparse_lambda = sparse_lambda
        self.ista_steps = ista_steps
        self.dictionary_steps = dictionary_steps
        self.max_training_windows = max_training_windows
        self.ridge_alpha = ridge_alpha
        self.dictionary = None
        self.readout = None

    def _initialise_dictionary(self, inputs: np.ndarray) -> np.ndarray:
        sample_size = min(len(inputs), self.n_atoms)
        indices = np.linspace(0, len(inputs) - 1, num=sample_size, dtype=int)
        dictionary = inputs[indices].T
        if dictionary.shape[1] < self.n_atoms:
            repeats = math.ceil(self.n_atoms / dictionary.shape[1])
            dictionary = np.tile(dictionary, repeats)[:, : self.n_atoms]
        norms = np.linalg.norm(dictionary, axis=0, keepdims=True) + 1e-8
        return dictionary / norms

    def _ista(self, inputs: np.ndarray, dictionary: np.ndarray) -> np.ndarray:
        gram = dictionary.T @ dictionary
        lipschitz = np.linalg.norm(gram, ord=2)
        step = 1.0 / (lipschitz + 1e-6)
        codes = np.zeros((inputs.shape[0], dictionary.shape[1]), dtype=np.float32)
        projected = inputs @ dictionary

        for _ in range(self.ista_steps):
            gradient = codes @ gram - projected
            codes = _soft_threshold(codes - step * gradient, self.sparse_lambda * step)
        return codes

    def fit(self, aggregate_windows: np.ndarray, target_windows: np.ndarray) -> None:
        aggregate_windows = _ensure_2d(aggregate_windows)
        target_windows = _ensure_2d(target_windows)
        sample_size = min(len(aggregate_windows), self.max_training_windows)
        sample_indices = np.linspace(0, len(aggregate_windows) - 1, num=sample_size, dtype=int)
        x_train = aggregate_windows[sample_indices]
        y_train = target_windows[sample_indices]

        dictionary = self._initialise_dictionary(x_train)
        for _ in range(self.dictionary_steps):
            codes = self._ista(x_train, dictionary)
            gram = codes.T @ codes + self.ridge_alpha * np.eye(self.n_atoms, dtype=np.float32)
            dictionary = (x_train.T @ codes) @ np.linalg.inv(gram)
            norms = np.linalg.norm(dictionary, axis=0, keepdims=True) + 1e-8
            dictionary = dictionary / norms

        codes = self._ista(x_train, dictionary)
        gram = codes.T @ codes + self.ridge_alpha * np.eye(self.n_atoms, dtype=np.float32)
        self.dictionary = dictionary.astype(np.float32)
        self.readout = (np.linalg.inv(gram) @ codes.T @ y_train).T.astype(np.float32)

    def disaggregate(self, inputs: np.ndarray) -> np.ndarray:
        if self.dictionary is None or self.readout is None:
            raise RuntimeError("DiscriminativeSparseCoding must be fitted before disaggregation.")
        inputs = _ensure_2d(inputs)
        codes = self._ista(inputs, self.dictionary)
        return codes @ self.readout.T

    def get_state(self) -> dict[str, Any]:
        return self._serialise_arrays({"dictionary": self.dictionary, "readout": self.readout})

    def set_state(self, state: dict[str, Any]) -> None:
        state = self._deserialise_arrays(state)
        self.dictionary = state["dictionary"]
        self.readout = state["readout"]
