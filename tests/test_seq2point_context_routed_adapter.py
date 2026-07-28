from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import torch

from experiments.run_seq2point_context_routed_adapter import (
    HouseholdRoutedTrainerModel,
)
from model_pipeline.context_data_feeder import (
    HouseholdUniformBatchSampler,
    MultiHouseQueryDataset,
    NormalisationStats,
    QueryWindowDataset,
)
from model_pipeline.models.context_adapter.seq2point_adapter_bank import (
    ContextRoutedAdapterBankSeq2Point,
    GlobalSharedAdapterSeq2Point,
)
from model_pipeline.models.context_adapter.seq2point_film import module_sha256
from model_pipeline.models.seq2point.seq2point import Seq2Point


def query_frame(house_index: int, length: int = 80) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": pd.date_range("2020-01-01", periods=length, freq="6s"),
            "aggregate": np.arange(length, dtype=np.float32) + house_index,
            "washing_machine": np.arange(length, dtype=np.float32),
            "segment_id": np.zeros(length, dtype=np.int64),
        }
    )


class ContextRoutedAdapterTest(unittest.TestCase):
    window_size = 59

    def setUp(self):
        torch.manual_seed(42)
        self.query = torch.randn(4, self.window_size)
        self.context = torch.randn(1, 4, self.window_size)
        self.mask = torch.ones(1, 4, dtype=torch.bool)

    def baseline(self) -> Seq2Point:
        return Seq2Point(window_size=self.window_size, hidden_dim=16).eval()

    def test_zero_initialized_adapters_are_exact_baseline_identity(self):
        baseline = self.baseline()
        global_adapter = GlobalSharedAdapterSeq2Point(
            baseline, bottleneck_dim=4, residual_scale=0.1
        ).eval()
        routed_adapter = ContextRoutedAdapterBankSeq2Point(
            self.baseline(),
            window_size=self.window_size,
            code_dim=8,
            bottleneck_dim=4,
        ).eval()
        routed_adapter.baseline.load_state_dict(baseline.state_dict())
        with torch.no_grad():
            expected = baseline(self.query)
            actual_global = global_adapter(self.query)
            actual_routed = routed_adapter(
                self.query, self.context, self.mask
            )
        torch.testing.assert_close(actual_global, expected, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(actual_routed, expected, rtol=1e-6, atol=1e-6)

    def test_router_starts_uniform_and_sums_to_one(self):
        adapter = ContextRoutedAdapterBankSeq2Point(
            self.baseline(),
            window_size=self.window_size,
            code_dim=8,
            bottleneck_dim=4,
        ).eval()
        with torch.no_grad():
            _, route = adapter.generate_route(self.context, self.mask)
        torch.testing.assert_close(
            route,
            torch.full_like(route, 0.5),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            route.sum(dim=-1),
            torch.ones(route.shape[0]),
            rtol=0,
            atol=1e-7,
        )

    def test_optimizer_step_never_changes_backbone(self):
        adapter = ContextRoutedAdapterBankSeq2Point(
            self.baseline(),
            window_size=self.window_size,
            code_dim=8,
            bottleneck_dim=4,
        )
        before = module_sha256(adapter.baseline)
        optimizer = torch.optim.AdamW(
            [parameter for parameter in adapter.parameters() if parameter.requires_grad],
            lr=0.001,
            weight_decay=0.0001,
        )
        loss = adapter(self.query, self.context, self.mask).square().mean()
        loss.backward()
        optimizer.step()
        self.assertEqual(before, module_sha256(adapter.baseline))
        self.assertFalse(
            any(parameter.requires_grad for parameter in adapter.baseline.parameters())
        )

    def test_uniform_house_sampler_uses_exactly_one_house_per_batch(self):
        stats = NormalisationStats(0.0, 1.0, 0.0, 1.0)
        combined = MultiHouseQueryDataset(
            [
                QueryWindowDataset(
                    query_frame(index),
                    house=house,
                    stats=stats,
                    window_size=11,
                )
                for index, house in enumerate(("H1", "H2", "H3", "H7"))
            ]
        )
        first = list(
            HouseholdUniformBatchSampler(
                combined, batch_size=8, seed=42, epoch_size=64
            )
        )
        second = list(
            HouseholdUniformBatchSampler(
                combined, batch_size=8, seed=42, epoch_size=64
            )
        )
        self.assertEqual(first, second)
        ranges = list(combined.house_ranges.values())
        for batch in first:
            containing_houses = [
                index
                for index, indices in enumerate(ranges)
                if set(batch).issubset(set(indices.tolist()))
            ]
            self.assertEqual(len(containing_houses), 1)

    def test_training_hook_rejects_cross_house_pairing(self):
        adapter = HouseholdRoutedTrainerModel(
            self.baseline(),
            window_size=self.window_size,
            code_dim=8,
            bottleneck_dim=4,
        )
        contexts = {
            house: (
                torch.randn(4, self.window_size),
                torch.ones(4, dtype=torch.bool),
            )
            for house in ("H1", "H2", "H5")
        }
        adapter.configure_training_contexts(
            contexts, ["H1", "H2"], "H5", torch.device("cpu")
        )
        inputs, targets = adapter.prepare_batch(
            (
                self.query,
                torch.randn(len(self.query)),
                torch.tensor([0, 1, 0, 1]),
            ),
            device="cpu",
        )
        with self.assertRaises(RuntimeError):
            adapter.compute_loss(inputs, targets, criterion=torch.nn.MSELoss())


if __name__ == "__main__":
    unittest.main()
