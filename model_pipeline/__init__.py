from model_pipeline.model_registry import (
    create_model,
    discover_models,
    get_model_entry,
    get_model_mappings,
    instantiate_from_checkpoint,
    list_models,
    load_checkpoint,
    register_model,
)

discover_models()

__all__ = [
    "create_model",
    "discover_models",
    "get_model_entry",
    "get_model_mappings",
    "instantiate_from_checkpoint",
    "list_models",
    "load_checkpoint",
    "register_model",
]
