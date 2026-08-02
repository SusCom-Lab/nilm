from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
import pkgutil
from typing import Any, Callable

import torch


@dataclass(frozen=True)
class ModelEntry:
    key: str
    cls: type
    display_name: str
    aliases: tuple[str, ...]
    family: str
    target_type: str
    supports_gradient: bool


_MODEL_REGISTRY: dict[str, ModelEntry] = {}
_MODEL_ALIASES: dict[str, str] = {}
_DISCOVERED = False


def _normalise_name(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def register_model(
    name: str,
    *,
    aliases: list[str] | tuple[str, ...] | None = None,
    display_name: str | None = None,
) -> Callable[[type], type]:
    aliases = tuple(aliases or ())

    def decorator(cls: type) -> type:
        key = _normalise_name(name)
        if key in _MODEL_REGISTRY:
            raise ValueError(f"Model '{name}' is already registered.")
        setattr(cls, "_registry_key", key)

        entry = ModelEntry(
            key=key,
            cls=cls,
            display_name=display_name or getattr(cls, "display_name", cls.__name__),
            aliases=aliases,
            family=getattr(cls, "model_family", "generic"),
            target_type=getattr(cls, "target_type", "point"),
            supports_gradient=bool(getattr(cls, "supports_gradient", False)),
        )
        _MODEL_REGISTRY[key] = entry

        all_names = (name, entry.display_name, *aliases)
        for alias in all_names:
            alias_key = _normalise_name(alias)
            if alias_key in _MODEL_ALIASES and _MODEL_ALIASES[alias_key] != key:
                raise ValueError(f"Alias '{alias}' is already mapped to another model.")
            _MODEL_ALIASES[alias_key] = key

        return cls

    return decorator


def discover_models() -> None:
    global _DISCOVERED
    if _DISCOVERED:
        return
    package = import_module("model_pipeline.models")
    for module in pkgutil.walk_packages(
        package.__path__,
        prefix=f"{package.__name__}.",
    ):
        import_module(module.name)
    _DISCOVERED = True


def _resolve_key(name: str) -> str:
    discover_models()
    alias_key = _normalise_name(name)
    if alias_key in _MODEL_ALIASES:
        return _MODEL_ALIASES[alias_key]
    raise KeyError(f"Unknown model '{name}'. Available models: {', '.join(sorted(_MODEL_REGISTRY))}")


def get_model_entry(name: str) -> ModelEntry:
    return _MODEL_REGISTRY[_resolve_key(name)]


def list_models() -> list[str]:
    discover_models()
    return [entry.display_name for entry in _MODEL_REGISTRY.values()]


def get_model_mappings() -> dict[str, type]:
    discover_models()
    return {entry.display_name: entry.cls for entry in _MODEL_REGISTRY.values()}


def create_model(name: str, **kwargs: Any) -> Any:
    entry = get_model_entry(name)
    return entry.cls(**kwargs)


def load_checkpoint(path: str, map_location: str | None = None) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def instantiate_from_checkpoint(checkpoint: dict[str, Any], map_location: str | None = None) -> Any:
    model_name = checkpoint.get("model_key") or checkpoint.get("model_name")
    if model_name is None:
        raise KeyError("Checkpoint is missing 'model_key' or 'model_name'.")

    init_kwargs = dict(checkpoint.get("init_kwargs", {}))
    if "window_size" not in init_kwargs and "window_length" in checkpoint:
        init_kwargs["window_size"] = checkpoint["window_length"]
    if "output_size" not in init_kwargs and "output_size" in checkpoint:
        init_kwargs["output_size"] = checkpoint["output_size"]
    if "output_offset" not in init_kwargs and "output_offset" in checkpoint:
        init_kwargs["output_offset"] = checkpoint["output_offset"]

    model = create_model(model_name, **init_kwargs)

    state = checkpoint.get("model_state")
    if state is None and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]

    if state is not None:
        if map_location is not None and hasattr(model, "load_exported_state"):
            model.load_exported_state(state, map_location=map_location)
        else:
            model.load_exported_state(state)

    return model
