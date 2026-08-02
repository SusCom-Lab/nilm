from __future__ import annotations

from abc import ABC, abstractmethod
import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn


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
