"""Official legacy AFHMM-SAC algorithm behind the repository plugin API.

Official source: https://github.com/nilmtk/nilmtk-contrib/blob/14efd545e09d836e02159b23444350af92c6ee70/nilmtk_contrib/disaggregate/afhmm_sac.py
Local adaptation: sequential blocks (instead of multiprocessing) and canonical partitions.
"""

from __future__ import annotations

from collections import Counter, OrderedDict
import math

import numpy as np

from model_pipeline.api import FitResult, InferenceContext, PredictionOutput, TrainingContext
from model_pipeline.model_registry import register_model
from model_pipeline.models.classical.afhmm import (
    AFHMMBaseline,
    _deserialise_value,
    _serialise_value,
)


@register_model("afhmm_sac", aliases=("AFHMM-SAC",), display_name="AFHMM_SAC")
class AFHMMSACBaseline(AFHMMBaseline):
    display_name = "AFHMM_SAC"

    def __init__(
        self,
        *,
        window_size: int = 720,
        n_states: int = 2,
        solver: str = "SCS",
        max_iters: int | None = None,
        eps: float | None = None,
        warm_start: bool = True,
        sac_strength: float = 1.0,
        optimisation_epochs: int = 6,
    ):
        super().__init__(window_size=window_size, n_states=n_states)
        self.time_period = window_size
        self.solver = solver
        self.max_iters = max_iters
        self.eps = eps
        self.warm_start = warm_start
        self.sac_strength = sac_strength
        self.optimisation_epochs = optimisation_epochs
        self.appliance_order: list[str] = []
        self.means_vector: dict[str, np.ndarray] = {}
        self.pi_vector: dict[str, np.ndarray] = {}
        self.transmat_vector: dict[str, np.ndarray] = {}
        self.signal_aggregates: dict[str, np.ndarray] = {}
        self._config = dict(
            window_size=window_size, n_states=n_states, solver=solver,
            max_iters=max_iters, eps=eps, warm_start=warm_start,
            sac_strength=sac_strength, optimisation_epochs=optimisation_epochs,
        )

    def fit(self, train_data, validation_data, context: TrainingContext) -> FitResult:
        del validation_data
        try:
            from hmmlearn import hmm
        except ImportError as exc:
            raise ImportError("AFHMM-SAC requires hmmlearn.") from exc
        names = tuple(train_data.series[0].appliances)
        self.appliance_order = list(names)
        self.means_vector, self.pi_vector = {}, {}
        self.transmat_vector, self.signal_aggregates = {}, {}
        for name in names:
            values = np.concatenate([
                series.appliance_power[:, series.appliances.index(name)]
                for series in train_data.series
            ]).reshape(-1, 1)
            model = hmm.GaussianHMM(self.n_states, "full")
            model.fit(values)
            means = model.means_.flatten().reshape(-1, 1)
            states = model.predict(values)
            counts = Counter(states.flatten())
            priors = np.asarray(
                [counts.get(index, 0) / len(states) for index in range(self.n_states)],
                dtype=np.float64,
            )
            # Positive clipping is an adapter-only numerical safeguard for log().
            self.means_vector[name] = means
            self.pi_vector[name] = np.clip(priors, 1e-12, None)
            self.transmat_vector[name] = np.clip(model.transmat_.T, 1e-12, None)
            self.signal_aggregates[name] = np.mean(values) * self.time_period
        validation = context.validate_candidate(epoch=1, model=self)
        return FitResult(
            history=[{"epoch": 1, "validation_mse": validation.mse}],
            best_epoch=1 if validation.improved else None,
            best_validation_mse=validation.mse,
            checkpoint_path=validation.checkpoint_path,
        )

    def _solve_problem(self, problem):
        kwargs = dict(solver=self.solver, verbose=False, warm_start=self.warm_start)
        if self.max_iters is not None:
            kwargs["max_iters"] = self.max_iters
        if self.eps is not None:
            kwargs["eps"] = self.eps
        return problem.solve(**kwargs)

    def _solve_block(self, test_mains):
        import cvxpy as cvx

        length = len(test_mains)
        sigma = 100 * np.ones((length, 1))
        states = None
        built = False
        for epoch in range(self.optimisation_epochs):
            if epoch % 2:
                if states is None:
                    raise RuntimeError("AFHMM-SAC solver did not return appliance states.")
                usage = sum(
                    np.sum(states[name] @ self.means_vector[name], axis=1)
                    for name in self.appliance_order
                )
                sigma = (test_mains.flatten() - usage.flatten()).reshape(-1, 1)
                sigma = np.where(sigma < 1, 1, sigma)
                continue
            if not built:
                constraints, state_variables, transition_variables = [], {}, {}
                for name in self.appliance_order:
                    state = cvx.Variable((length, self.n_states), name=f"state-{name}")
                    transitions = [
                        cvx.Variable((self.n_states, self.n_states), name=f"transition-{name}-{t}")
                        for t in range(length)
                    ]
                    state_variables[name], transition_variables[name] = state, transitions
                    constraints += [state >= 0, state <= 1]
                    for t in range(length):
                        constraints += [cvx.sum(state[t]) == 1, transitions[t] >= 0, transitions[t] <= 1]
                        for i in range(self.n_states):
                            constraints.append(cvx.sum(transitions[t].T[i]) == state[t][i])
                    for t in range(1, length):
                        for i in range(self.n_states):
                            constraints.append(cvx.sum(transitions[t][i]) == state[t - 1][i])
                    constraints.append(
                        cvx.sum(state @ self.means_vector[name])
                        <= self.sac_strength * self.signal_aggregates[name]
                    )
                total = sum(
                    state_variables[name] @ self.means_vector[name]
                    for name in self.appliance_order
                )
                transition_term = sum(
                    -cvx.sum(cvx.multiply(matrix, np.log(self.transmat_vector[name])))
                    for name in self.appliance_order
                    for matrix in transition_variables[name]
                )
                initial_term = sum(
                    -cvx.sum(cvx.multiply(state_variables[name][0], np.log(self.pi_vector[name])))
                    for name in self.appliance_order
                )
                built = True
            variance_term = sum(
                0.5 * ((test_mains[t, 0] - total[t, 0]) ** 2 / sigma[t, 0] ** 2)
                + 0.5 * np.log(sigma[t, 0] ** 2)
                for t in range(length)
            )
            problem = cvx.Problem(
                cvx.Minimize(transition_term + initial_term + variance_term), constraints
            )
            self._solve_problem(problem)
            states = {name: state_variables[name].value for name in self.appliance_order}
        if states is None or any(value is None for value in states.values()):
            raise RuntimeError("AFHMM-SAC optimisation failed.")
        return np.column_stack([
            np.sum(states[name] @ self.means_vector[name], axis=1)
            for name in self.appliance_order
        ]).astype(np.float32)

    def _predict_series(self, aggregate):
        if not self.appliance_order:
            raise RuntimeError("AFHMM-SAC must be fitted before prediction.")
        values = np.asarray(aggregate).reshape(-1, 1)
        blocks = [
            self._solve_block(values[start : start + self.time_period])
            for start in range(0, len(values), self.time_period)
        ]
        return np.concatenate(blocks, axis=0)[:len(values)]

    def predict(self, inference_data, context: InferenceContext) -> PredictionOutput:
        del context
        timestamps, households, powers = [], [], []
        for series in inference_data.series:
            predicted = self._predict_series(series.aggregate)
            order = [self.appliance_order.index(name) for name in series.appliances]
            timestamps.append(series.timestamps)
            households.append(np.repeat(series.household_id, len(series.timestamps)))
            powers.append(predicted[:, order])
        return PredictionOutput(
            timestamps=np.concatenate(timestamps), power=np.concatenate(powers),
            appliances=inference_data.series[0].appliances,
            household_ids=np.concatenate(households),
        )

    def get_state(self):
        return _serialise_value(dict(
            appliance_order=self.appliance_order,
            means_vector=self.means_vector,
            pi_vector=self.pi_vector,
            transmat_vector=self.transmat_vector,
            signal_aggregates=self.signal_aggregates,
        ))

    def set_state(self, state):
        restored = _deserialise_value(state)
        self.appliance_order = list(restored["appliance_order"])
        self.means_vector = {key: np.asarray(value) for key, value in restored["means_vector"].items()}
        self.pi_vector = {key: np.asarray(value) for key, value in restored["pi_vector"].items()}
        self.transmat_vector = {key: np.asarray(value) for key, value in restored["transmat_vector"].items()}
        self.signal_aggregates = {
            key: float(np.asarray(value)) for key, value in restored["signal_aggregates"].items()
        }
