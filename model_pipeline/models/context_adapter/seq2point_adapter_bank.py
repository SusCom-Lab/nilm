"""Identity-initialized residual adapters for a frozen Seq2Point backbone."""

from __future__ import annotations

import torch
import torch.nn as nn

from model_pipeline.models.context_adapter.context_encoder import ContextEncoder
from model_pipeline.models.context_adapter.seq2point_film import _FrozenSeq2PointAdapter
from model_pipeline.models.seq2point.seq2point import Seq2Point


class BottleneckAdapter(nn.Module):
    """A 1024 -> bottleneck -> 1024 residual branch with a zero output layer."""

    def __init__(self, hidden_dim: int, bottleneck_dim: int):
        super().__init__()
        self.down = nn.Linear(hidden_dim, bottleneck_dim)
        self.activation = nn.ReLU()
        self.up = nn.Linear(bottleneck_dim, hidden_dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.up(self.activation(self.down(hidden)))


class GlobalSharedAdapterSeq2Point(_FrozenSeq2PointAdapter):
    """One context-free residual adapter inserted before the original decoder."""

    def __init__(
        self,
        baseline: Seq2Point,
        *,
        bottleneck_dim: int = 64,
        residual_scale: float = 0.1,
    ):
        super().__init__(baseline)
        self.residual_scale = float(residual_scale)
        self.adapter = BottleneckAdapter(self.hidden_dim, bottleneck_dim)

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        self.assert_baseline_frozen()
        hidden = self.baseline.encode(query)
        return self.baseline.decode(
            hidden + self.residual_scale * self.adapter(hidden)
        )


class ContextRoutedAdapterBankSeq2Point(_FrozenSeq2PointAdapter):
    """Route a household-fixed mixture over residual bottleneck adapters."""

    def __init__(
        self,
        baseline: Seq2Point,
        *,
        window_size: int = 599,
        code_dim: int = 128,
        num_adapters: int = 2,
        bottleneck_dim: int = 64,
        residual_scale: float = 0.1,
        router_temperature: float = 1.0,
    ):
        super().__init__(baseline)
        if int(num_adapters) != 2:
            raise ValueError("The development protocol fixes num_adapters=2.")
        if float(router_temperature) <= 0:
            raise ValueError("router_temperature must be positive.")
        self.num_adapters = int(num_adapters)
        self.residual_scale = float(residual_scale)
        self.router_temperature = float(router_temperature)
        self.context_encoder = ContextEncoder(window_size=window_size, code_dim=code_dim)
        self.router = nn.Linear(code_dim, self.num_adapters)
        nn.init.zeros_(self.router.weight)
        nn.init.zeros_(self.router.bias)
        self.adapters = nn.ModuleList(
            BottleneckAdapter(self.hidden_dim, bottleneck_dim)
            for _ in range(self.num_adapters)
        )

    def generate_route(
        self,
        context_windows: torch.Tensor,
        context_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        code = self.context_encoder(context_windows, context_mask)
        route = torch.softmax(self.router(code) / self.router_temperature, dim=-1)
        return code, route

    def adapt_hidden(
        self,
        hidden: torch.Tensor,
        route: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if route.ndim == 1:
            route = route.unsqueeze(0)
        if route.ndim != 2 or route.shape[-1] != self.num_adapters:
            raise ValueError(
                f"Expected route [..., {self.num_adapters}], got {tuple(route.shape)}."
            )
        outputs = torch.stack(
            [adapter(hidden) for adapter in self.adapters],
            dim=1,
        )
        residual = torch.sum(outputs * route.unsqueeze(-1), dim=1)
        return hidden + self.residual_scale * residual, residual

    def forward_with_route(
        self,
        query: torch.Tensor,
        route: torch.Tensor,
    ) -> torch.Tensor:
        self.assert_baseline_frozen()
        hidden = self.baseline.encode(query)
        adapted, _ = self.adapt_hidden(hidden, route)
        return self.baseline.decode(adapted)

    def forward(
        self,
        query: torch.Tensor,
        context_windows: torch.Tensor,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _, route = self.generate_route(context_windows, context_mask)
        return self.forward_with_route(query, route)
