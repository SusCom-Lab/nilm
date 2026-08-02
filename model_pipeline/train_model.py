"""Public experiment runner: fixed splits, validation MSE, and checkpoints only."""

from __future__ import annotations

import os
import random
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from model_pipeline.api import (
    InferenceContext,
    OFFICIAL_EXPERIMENT_SEEDS,
    TrainingContext,
    ValidationResult,
)
from model_pipeline.data_protocol import (
    build_evaluation_target,
    hide_partition_targets,
    load_supervised_partition,
    same_csv_dataset,
    split_supervised_partition,
    validate_experiment_partitions,
)
from model_pipeline.model_registry import create_model, load_checkpoint


def set_random_seed(seed: int) -> None:
    if seed not in OFFICIAL_EXPERIMENT_SEEDS:
        raise ValueError(f"Experiment seed must be one of {OFFICIAL_EXPERIMENT_SEEDS}.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _aligned_prediction(prediction, target, aggregate: np.ndarray) -> np.ndarray:
    if prediction.appliances != target.appliances:
        raise ValueError("Prediction appliance ordering does not match the public protocol.")
    if not np.array_equal(prediction.household_ids, target.household_ids):
        raise ValueError("Prediction households do not match the complete public timeline.")
    if not pd.Index(prediction.timestamps).equals(pd.Index(target.timestamps)):
        raise ValueError("Prediction timestamps do not match the complete public timeline.")
    if prediction.power.shape != target.appliance_power.shape:
        raise ValueError("Prediction shape does not match the public evaluation target.")
    ceiling = np.repeat(np.clip(aggregate, 0.0, None)[:, None], prediction.power.shape[1], axis=1)
    return np.minimum(np.clip(prediction.power, 0.0, None), ceiling)


def validation_mse(prediction, target, aggregate: np.ndarray) -> float:
    power = _aligned_prediction(prediction, target, aggregate)
    truth = np.clip(target.appliance_power, 0.0, None)
    return float(np.mean(np.square(power - truth), dtype=np.float64))


class Trainer:
    """Run one fixed-seed experiment without owning any model training logic."""

    def __init__(
        self,
        *,
        model_name: str,
        train_csv_dirs: list[str],
        validation_csv_dirs: list[str],
        appliance: str,
        dataset: str,
        model_save_dir: str | None = None,
        result_dir: str | None = None,
        seed: int = 42,
        crop: int | None = None,
        device: str | None = None,
        patience: int = 8,
        min_delta: float = 1e-4,
        model=None,
        model_init_kwargs: dict[str, Any] | None = None,
        window_length: int | None = None,
        **legacy_options: Any,
    ) -> None:
        if legacy_options:
            unknown = ", ".join(sorted(legacy_options))
            raise TypeError(f"Unsupported Trainer options: {unknown}.")
        set_random_seed(seed)
        init_kwargs = dict(model_init_kwargs or {})
        if window_length is not None:
            init_kwargs.setdefault("window_size", int(window_length))
        self.model = model or create_model(model_name, **init_kwargs)
        self.model_init_kwargs = init_kwargs
        self.model_name = getattr(self.model, "display_name", model_name)
        self.appliance = appliance
        self.appliance_name_formatted = appliance.replace(" ", "_")
        self.dataset = dataset
        self.seed = seed
        self.crop = crop
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.model_save_dir = model_save_dir or os.path.join(os.getcwd(), "saved_models")
        self.result_dir = result_dir or os.path.join(
            "result", dataset.lower(), self.appliance_name_formatted
        )
        os.makedirs(self.model_save_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)

        self.shared_dataset_split = same_csv_dataset(
            train_csv_dirs, validation_csv_dirs
        )
        if self.shared_dataset_split:
            complete_data = load_supervised_partition(
                "training_validation", train_csv_dirs, crop=crop
            )
            self.train_data, self.validation_supervised = split_supervised_partition(
                complete_data, train_ratio=0.8
            )
        else:
            self.train_data = load_supervised_partition(
                "training", train_csv_dirs, crop=crop
            )
            self.validation_supervised = load_supervised_partition(
                "validation", validation_csv_dirs, crop=crop
            )
            validate_experiment_partitions(self.train_data, self.validation_supervised)
        self.validation_input = hide_partition_targets(self.validation_supervised)
        self.validation_target = build_evaluation_target(self.validation_supervised)
        self.validation_aggregate = np.concatenate(
            [series.aggregate for series in self.validation_supervised.series]
        )
        self.checkpoint_path = os.path.join(
            self.model_save_dir,
            f"{self.appliance}_{self.dataset}_{self.model_name}_seed{self.seed}.pth",
        )
        self.best_val_loss = float("inf")
        self.best_epoch = None
        self.counter = 0
        self.epoch_history: list[dict[str, Any]] = []

    def _metadata(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "appliance": self.appliance,
            "seed": self.seed,
            "train_households": list(self.train_data.household_ids),
            "validation_households": list(self.validation_supervised.household_ids),
            "split_strategy": (
                "chronological_80_20" if self.shared_dataset_split else "disjoint_households"
            ),
            "model_config": dict(getattr(self.model, "_config", self.model_init_kwargs)),
            "num_epochs": self.num_epochs,
        }

    def _validate_candidate(self, *, epoch: int, model, **_ignored) -> ValidationResult:
        prediction = model.predict(
            self.validation_input,
            InferenceContext(
                device=self.device,
                batch_size=int(getattr(model, "official_batch_size", 512)),
            ),
        )
        mse = validation_mse(prediction, self.validation_target, self.validation_aggregate)
        improved = mse < self.best_val_loss - self.min_delta
        checkpoint_path = None
        if improved:
            self.best_val_loss = mse
            self.best_epoch = int(epoch)
            self.counter = 0
            model.save(self.checkpoint_path, metadata=self._metadata())
            checkpoint_path = self.checkpoint_path
        else:
            self.counter += 1
        return ValidationResult(
            epoch=int(epoch),
            mse=mse,
            improved=improved,
            should_stop=self.counter >= self.patience,
            checkpoint_path=checkpoint_path,
        )

    def trainModel(self, num_epochs: int | None = None):
        official_epochs = int(getattr(self.model, "default_num_epochs", 10))
        self.num_epochs = official_epochs if num_epochs is None else int(num_epochs)
        if self.num_epochs < 1:
            raise ValueError("num_epochs must be at least 1.")
        context = TrainingContext(
            device=self.device,
            seed=self.seed,
            num_epochs=self.num_epochs,
            validate_candidate=self._validate_candidate,
            metadata={
                "dataset": self.dataset,
                "appliance": self.appliance,
                "split_strategy": (
                    "chronological_80_20"
                    if self.shared_dataset_split
                    else "disjoint_households"
                ),
            },
        )
        result = self.model.fit(self.train_data, self.validation_input, context)
        self.epoch_history = list(result.history)
        if self.best_epoch is None:
            raise RuntimeError("Training finished without a public-validation checkpoint.")
        self._save_history()
        return result

    def _save_history(self) -> None:
        path = os.path.join(
            self.result_dir, f"{self.model_name}_seed{self.seed}_training_history.csv"
        )
        pd.DataFrame(self.epoch_history).to_csv(path, index=False)

    def plotLosses(self) -> None:
        if not self.epoch_history:
            return
        frame = pd.DataFrame(self.epoch_history)
        plt.figure(figsize=(8, 5))
        for column in ("training_loss", "validation_mse"):
            if column in frame:
                plt.plot(frame["epoch"], frame[column], label=column)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.result_dir, f"{self.model_name}_seed{self.seed}_loss.png"),
            dpi=150,
        )
        plt.close()


class Finetuner:
    """Compatibility wrapper; fine-tuning remains outside fair benchmark runs."""

    def __init__(self, model_state_dir: str, *, validation_csv_dirs=None, **kwargs):
        checkpoint = load_checkpoint(model_state_dir, map_location="cpu")
        model = create_model(checkpoint["model_key"], **checkpoint.get("init_kwargs", {}))
        model.load(model_state_dir, kwargs.get("device") or "cpu")
        if validation_csv_dirs is None:
            raise ValueError("Fine-tuning requires separately specified validation households.")
        finetune_csv = kwargs.pop("finetune_csv_dir")
        self.trainer = Trainer(
            model=model,
            model_name=checkpoint["model_key"],
            train_csv_dirs=[finetune_csv],
            validation_csv_dirs=validation_csv_dirs,
            appliance=checkpoint.get("metadata", {}).get("appliance", "appliance"),
            **kwargs,
        )

    def fineTune(self, max_epochs=None): return self.trainer.trainModel(max_epochs)
    def plotLosses(self): return self.trainer.plotLosses()


__all__ = [
    "Finetuner",
    "Trainer",
    "OFFICIAL_EXPERIMENT_SEEDS",
    "set_random_seed",
    "validation_mse",
]
