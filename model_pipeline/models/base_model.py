from __future__ import annotations

from abc import ABC, abstractmethod
import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from model_pipeline.contracts import FitResult, PredictionOutput


class BaseNILMModel(ABC):
    display_name = "Base NILM Model"
    model_family = "generic"
    target_type = "point"
    supports_gradient = False
    default_window_size = 599
    default_output_size = 1

    def __init__(
        self,
        *,
        window_size: int | None = None,
        output_size: int | None = None,
        output_offset: int | None = None,
        **config: Any,
    ) -> None:
        self.window_size = int(window_size or self.default_window_size)
        default_output_size = self.default_output_size
        if default_output_size is None:
            default_output_size = self.window_size if self.target_type == "sequence" else 1
        self.output_size = int(output_size or default_output_size)
        self.output_offset = int(self._default_output_offset() if output_offset is None else output_offset)
        self._config = {
            "window_size": self.window_size,
            "output_size": self.output_size,
            "output_offset": self.output_offset,
            **config,
        }

    def _default_output_offset(self) -> int:
        if self.target_type == "point":
            return self.window_size // 2
        return 0

    def get_window_size(self) -> int:
        return self.window_size

    def getWindowSize(self) -> int:
        return self.get_window_size()

    def get_output_size(self) -> int:
        return self.output_size

    def get_output_offset(self) -> int:
        return self.output_offset

    def get_target_type(self) -> str:
        return self.target_type

    def get_init_kwargs(self) -> dict[str, Any]:
        return dict(self._config)

    def fit(self, train_data, validation_data, context):
        """Train this model using its private, source-faithful recipe.

        Concrete plugins own their Dataset, loss, optimiser, scheduler, and
        epoch loop.  The experiment runner only supplies fixed household
        partitions and checkpoint services through ``context``.
        """

        del train_data, validation_data, context
        raise NotImplementedError(f"{self.__class__.__name__} must implement fit().")

    def predict(self, inference_data, context):
        """Return a timestamp-aligned ``PredictionOutput`` for inference data."""

        del inference_data, context
        raise NotImplementedError(f"{self.__class__.__name__} must implement predict().")

    def save(self, path: str | os.PathLike[str]) -> str:
        """Persist the model-owned state without changing repository paths."""

        path = os.fspath(path)
        torch.save(self.export_state(), path)
        return path

    def load(self, path: str | os.PathLike[str], map_location: str | None = None) -> None:
        """Load state saved by :meth:`save` or a repository checkpoint."""

        path = os.fspath(path)
        try:
            payload = torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location=map_location)
        if isinstance(payload, dict):
            payload = payload.get("model_state", payload.get("model_state_dict", payload))
        self.load_exported_state(payload, map_location=map_location)

    def prepare_targets(self, targets: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        if isinstance(targets, torch.Tensor):
            targets = targets.float()
            if self.target_type == "point":
                return targets.reshape(-1)
            if targets.ndim == 3 and targets.size(-1) == 1:
                targets = targets.squeeze(-1)
            return targets

        targets = np.asarray(targets, dtype=np.float32)
        if self.target_type == "point":
            return targets.reshape(-1)
        if targets.ndim == 3 and targets.shape[-1] == 1:
            targets = np.squeeze(targets, axis=-1)
        return targets

    def prepare_outputs(self, outputs: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        if isinstance(outputs, torch.Tensor):
            outputs = outputs.float()
            if self.target_type == "point":
                return outputs.reshape(-1)
            if outputs.ndim == 3 and outputs.size(-1) == 1:
                outputs = outputs.squeeze(-1)
            return outputs

        outputs = np.asarray(outputs, dtype=np.float32)
        if self.target_type == "point":
            return outputs.reshape(-1)
        if outputs.ndim == 3 and outputs.shape[-1] == 1:
            outputs = np.squeeze(outputs, axis=-1)
        return outputs

    @abstractmethod
    def disaggregate(self, inputs: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def export_state(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def load_exported_state(self, state: Any, map_location: str | None = None) -> None:
        raise NotImplementedError


class TorchNILMModel(nn.Module, BaseNILMModel):
    supports_gradient = True

    def __init__(self, **kwargs: Any) -> None:
        nn.Module.__init__(self)
        BaseNILMModel.__init__(self, **kwargs)

    def disaggregate(self, inputs: torch.Tensor | np.ndarray) -> torch.Tensor:
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.as_tensor(inputs, dtype=torch.float32)
        return self(inputs)

    def _prepare_plugin_batch(self, batch, device: str):
        hook = getattr(self, "prepare_batch", None)
        if callable(hook):
            return hook(batch, device=device)
        if not isinstance(batch, (tuple, list)) or len(batch) != 2:
            raise ValueError(
                "Model batches must be (inputs, targets), unless prepare_batch is implemented."
            )
        inputs, targets = batch
        return inputs.to(device), self.prepare_targets(targets.to(device))

    def _plugin_loss(self, inputs, targets, criterion):
        hook = getattr(self, "compute_loss", None)
        if callable(hook):
            result = hook(inputs, targets, criterion=criterion)
            return result[0] if isinstance(result, tuple) else result
        outputs = self.prepare_outputs(self(inputs))
        return criterion(outputs, targets)

    def _plugin_validation(self, loader, device: str, criterion):
        if loader is None:
            return None, None
        self.eval()
        validation_loss = 0.0
        selection_loss = 0.0
        selection_hook = getattr(self, "compute_selection_loss", None)
        with torch.no_grad():
            for batch in loader:
                inputs, targets = self._prepare_plugin_batch(batch, device)
                loss = self._plugin_loss(inputs, targets, criterion)
                validation_loss += float(loss.item())
                selected = (
                    selection_hook(inputs, targets, criterion=criterion)
                    if callable(selection_hook)
                    else loss
                )
                selection_loss += float(selected.item())
        count = max(1, len(loader))
        return validation_loss / count, selection_loss / count

    def fit(self, train_data, validation_data, context) -> FitResult:
        """Default recipe for simple gradient-based baselines.

        Models with a source-specific Dataset, mask, optimiser, scheduler, or
        multi-stage objective override this method in their own model file.
        """

        train_loader = getattr(train_data, "loader", None)
        validation_loader = getattr(validation_data, "loader", None)
        if train_loader is None:
            raise ValueError(f"{self.__class__.__name__} requires a training DataLoader.")

        device = context.device
        self.to(device)
        config = dict(context.config)
        criterion = config.get("criterion") or nn.MSELoss()
        optimizer = config.get("optimizer") or optim.Adam(
            self.parameters(), lr=0.001, betas=(0.9, 0.999)
        )
        scheduler = config.get("scheduler") or ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2, threshold=1e-4
        )
        gradient_clip_norm = config.get("gradient_clip_norm")
        select_epoch_zero = bool(config.get("select_epoch_zero", False))
        history = []
        best_epoch = None
        best_score = None
        checkpoint_path = None

        def report(epoch, train_loss, validation_loss, selection_loss):
            nonlocal best_epoch, best_score, checkpoint_path
            record = {
                "epoch": int(epoch),
                "train_mse": train_loss,
                "validation_mse": validation_loss,
                "selection_loss": selection_loss,
                "checkpoint_eligible": True,
            }
            history.append(record)
            decision = context.checkpoint_callback(
                model=self,
                epoch=int(epoch),
                train_loss=train_loss,
                validation_loss=validation_loss,
                selection_loss=selection_loss,
                record=record,
            ) or {}
            if decision.get("saved"):
                best_epoch = int(epoch)
                best_score = float(selection_loss)
                checkpoint_path = decision.get("checkpoint_path")
            return bool(decision.get("should_stop", False))

        if select_epoch_zero and validation_loader is not None:
            validation_loss, selection_loss = self._plugin_validation(
                validation_loader, device, criterion
            )
            report(0, None, validation_loss, selection_loss)

        for epoch in range(1, int(context.num_epochs) + 1):
            batch_sampler = getattr(train_loader, "batch_sampler", None)
            set_epoch = getattr(batch_sampler, "set_epoch", None)
            if callable(set_epoch):
                set_epoch(epoch)

            self.train()
            total = 0.0
            for batch in train_loader:
                inputs, targets = self._prepare_plugin_batch(batch, device)
                optimizer.zero_grad()
                loss = self._plugin_loss(inputs, targets, criterion)
                loss.backward()
                if gradient_clip_norm is not None:
                    nn.utils.clip_grad_norm_(
                        [parameter for parameter in self.parameters() if parameter.requires_grad],
                        float(gradient_clip_norm),
                    )
                optimizer.step()
                total += float(loss.item())

            train_loss = total / max(1, len(train_loader))
            validation_loss, selection_loss = self._plugin_validation(
                validation_loader, device, criterion
            )
            if validation_loss is None:
                validation_loss = train_loss
                selection_loss = train_loss
            scheduler.step(selection_loss)
            if report(epoch, train_loss, validation_loss, selection_loss):
                break

        return FitResult(
            best_epoch=best_epoch,
            best_score=best_score,
            history=history,
            checkpoint_path=checkpoint_path,
        )

    def predict(self, inference_data, context) -> PredictionOutput:
        """Default aggregate-only inference for point and sequence baselines."""

        from model_pipeline.data_feeder import reconstruct_series_from_windows
        from model_pipeline.inference_data import InferenceWindowDataset

        stats = context.normalisation_stats
        if stats is None:
            raise ValueError("Model inference requires training normalisation_stats.")
        dataset = InferenceWindowDataset(
            timestamps=inference_data.timestamps,
            aggregate=inference_data.aggregate,
            segment_ids=inference_data.segment_ids,
            window_size=self.get_window_size(),
            output_size=self.get_output_size(),
            output_offset=self.get_output_offset(),
            normalisation_stats=stats,
            include_temporal_features=bool(
                getattr(self, "requires_temporal_features", False)
            ),
        )
        if len(dataset) == 0:
            raise ValueError("Inference data does not contain a complete model window.")

        loader = DataLoader(dataset, batch_size=context.batch_size, shuffle=False)
        self.to(context.device)
        self.eval()
        windows = []
        with torch.inference_mode():
            for inputs in loader:
                outputs = self.prepare_outputs(self(inputs.to(context.device)))
                windows.append(outputs.detach().cpu().numpy())

        window_predictions = np.concatenate(windows, axis=0)
        window_predictions = (
            window_predictions * float(stats["appliance_std"])
            + float(stats["appliance_mean"])
        )
        power, coverage = reconstruct_series_from_windows(
            window_predictions,
            len(inference_data.timestamps),
            target_mode=self.get_target_type(),
            output_offset=self.get_output_offset(),
            output_size=self.get_output_size(),
            window_start_indices=dataset.window_start_indices,
        )
        return PredictionOutput(
            timestamps=np.asarray(inference_data.timestamps),
            power=power,
            valid_mask=coverage > 0,
            metadata={"coverage": coverage},
        )

    def export_state(self) -> Any:
        return self.state_dict()

    def load_exported_state(self, state: Any, map_location: str | None = None) -> None:
        del map_location
        self.load_state_dict(state)

    def freeze_for_finetuning(self) -> None:
        for param in self.parameters():
            param.requires_grad = False
        for module in self.modules():
            if isinstance(module, nn.Linear):
                for param in module.parameters():
                    param.requires_grad = True


class ClassicalNILMModel(BaseNILMModel):
    supports_gradient = False
    is_joint_classical = False

    def is_joint_model(self) -> bool:
        return False

    def fit(self, aggregate_windows: np.ndarray, target_windows: np.ndarray) -> None:
        raise NotImplementedError(f"{self.__class__.__name__} must implement fit().")

    def export_state(self) -> Any:
        return self.get_state()

    def load_exported_state(self, state: Any, map_location: str | None = None) -> None:
        del map_location
        self.set_state(state)

    @abstractmethod
    def get_state(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def set_state(self, state: dict[str, Any]) -> None:
        raise NotImplementedError


class JointClassicalNILMModel(ClassicalNILMModel):
    target_type = "sequence"
    default_output_size = None
    is_joint_classical = True

    def get_grouping_key(self) -> str:
        return "house"

    def is_joint_model(self) -> bool:
        return True

    def fit_joint(self, grouped_train_data: dict[str, dict[str, Any]]) -> None:
        raise NotImplementedError(f"{self.__class__.__name__} must implement fit_joint().")

    def disaggregate_joint(self, grouped_test_data: dict[str, dict[str, Any]]) -> dict[str, Any]:
        raise NotImplementedError(f"{self.__class__.__name__} must implement disaggregate_joint().")
