"""Identity-initialized FiLM adapters for a frozen Seq2Point baseline."""

from __future__ import annotations

import hashlib
import torch
import torch.nn as nn

from model_pipeline.models.context_adapter.context_encoder import ContextEncoder
from model_pipeline.models.seq2point.seq2point import Seq2Point


def module_sha256(module: nn.Module) -> str:
    """Stable hash of tensor values in a module state dict."""
    digest = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def freeze_seq2point(baseline: Seq2Point) -> Seq2Point:
    for parameter in baseline.parameters():
        parameter.requires_grad = False
    baseline.eval()
    return baseline


def _bounded(raw: torch.Tensor) -> torch.Tensor:
    return 0.1 * torch.tanh(raw)


class _FrozenSeq2PointAdapter(nn.Module):
    supports_gradient = True

    def __init__(self, baseline: Seq2Point):
        super().__init__()
        if not isinstance(baseline, Seq2Point):
            raise TypeError("FiLM adapters require the repository Seq2Point implementation.")
        self.baseline = freeze_seq2point(baseline)
        self.hidden_dim = int(baseline.get_init_kwargs()["hidden_dim"])

    def train(self, mode: bool = True):
        super().train(mode)
        self.baseline.eval()
        return self

    def assert_baseline_frozen(self) -> None:
        if any(parameter.requires_grad for parameter in self.baseline.parameters()):
            raise RuntimeError("Seq2Point must remain frozen while training a FiLM adapter.")

    def get_window_size(self) -> int:
        return self.baseline.get_window_size()

    def get_output_size(self) -> int:
        return self.baseline.get_output_size()

    def get_output_offset(self) -> int:
        return self.baseline.get_output_offset()

    def get_target_type(self) -> str:
        return self.baseline.get_target_type()

    def get_init_kwargs(self) -> dict:
        return self.baseline.get_init_kwargs()

    def prepare_targets(self, targets: torch.Tensor) -> torch.Tensor:
        return self.baseline.prepare_targets(targets)

    def prepare_outputs(self, outputs: torch.Tensor) -> torch.Tensor:
        return self.baseline.prepare_outputs(outputs)

    def _apply_film(
        self,
        query: torch.Tensor,
        gamma_raw: torch.Tensor,
        beta_raw: torch.Tensor,
    ) -> torch.Tensor:
        self.assert_baseline_frozen()
        hidden = self.baseline.encode(query)
        gamma = _bounded(gamma_raw)
        beta = _bounded(beta_raw)
        if gamma.ndim == 1:
            gamma = gamma.unsqueeze(0)
            beta = beta.unsqueeze(0)
        return self.baseline.decode(hidden * (1.0 + gamma) + beta)


class GlobalFiLMSeq2Point(_FrozenSeq2PointAdapter):
    """One shared FiLM vector; it never consumes household context."""

    def __init__(self, baseline: Seq2Point):
        super().__init__(baseline)
        self.gamma_global_raw = nn.Parameter(torch.zeros(self.hidden_dim))
        self.beta_global_raw = nn.Parameter(torch.zeros(self.hidden_dim))

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        return self._apply_film(query, self.gamma_global_raw, self.beta_global_raw)

    def film_parameters(self) -> tuple[torch.Tensor, torch.Tensor]:
        return _bounded(self.gamma_global_raw), _bounded(self.beta_global_raw)


class ContextFiLMSeq2Point(_FrozenSeq2PointAdapter):
    """Generate household FiLM vectors from aggregate-only context windows."""

    def __init__(
        self,
        baseline: Seq2Point,
        *,
        window_size: int = 599,
        code_dim: int = 128,
        generator_hidden_dim: int = 256,
    ):
        super().__init__(baseline)
        self.context_encoder = ContextEncoder(window_size=window_size, code_dim=code_dim)
        self.generator = nn.Sequential(
            nn.Linear(code_dim, generator_hidden_dim),
            nn.ReLU(),
            nn.Linear(generator_hidden_dim, 2 * self.hidden_dim),
        )
        nn.init.zeros_(self.generator[-1].weight)
        nn.init.zeros_(self.generator[-1].bias)

    def generate_film(
        self,
        context_windows: torch.Tensor,
        context_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        code = self.context_encoder(context_windows, context_mask)
        gamma_raw, beta_raw = self.generator(code).chunk(2, dim=-1)
        return code, _bounded(gamma_raw), _bounded(beta_raw)

    def forward(
        self,
        query: torch.Tensor,
        context_windows: torch.Tensor,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _, gamma, beta = self.generate_film(context_windows, context_mask)
        return self.forward_with_film(query, gamma, beta)

    def forward_with_film(
        self,
        query: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        """Apply already bounded FiLM vectors without regenerating context."""
        self.assert_baseline_frozen()
        hidden = self.baseline.encode(query)
        if gamma.ndim == 1:
            gamma = gamma.unsqueeze(0)
            beta = beta.unsqueeze(0)
        return self.baseline.decode(hidden * (1.0 + gamma) + beta)
