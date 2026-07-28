"""Inference-side adapters for frozen NILM models."""

from model_pipeline.models.context_adapter.context_encoder import ContextEncoder
from model_pipeline.models.context_adapter.seq2point_film import (
    ContextFiLMSeq2Point,
    GlobalFiLMSeq2Point,
    freeze_seq2point,
    module_sha256,
)

__all__ = [
    "ContextEncoder",
    "ContextFiLMSeq2Point",
    "GlobalFiLMSeq2Point",
    "freeze_seq2point",
    "module_sha256",
]
