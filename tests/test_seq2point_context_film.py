from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import torch

from experiments.run_seq2point_context_film import (
    is_trained_baseline_checkpoint_improvement,
)
from model_pipeline.context_data_feeder import (
    HouseholdBalancedBatchSampler,
    LeakageGuard,
    MultiHouseQueryDataset,
    NormalisationStats,
    QueryWindowDataset,
    assert_common_evaluation_timestamps,
    select_uniform_context_windows,
)
from model_pipeline.context_data_feeder import BlockSpec
from model_pipeline.models.context_adapter.seq2point_film import (
    ContextFiLMSeq2Point,
    GlobalFiLMSeq2Point,
    module_sha256,
)
from model_pipeline.models.seq2point.seq2point import Seq2Point


def aggregate_frame(length: int = 100, *, with_labels: bool = False) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "time": pd.date_range("2020-01-01", periods=length, freq="6s"),
            "aggregate": np.linspace(100.0, 300.0, length, dtype=np.float32),
            "segment_id": np.zeros(length, dtype=np.int64),
        }
    )
    if with_labels:
        frame["washing_machine"] = np.arange(length, dtype=np.float32)
        frame["status"] = 0
    return frame


class Seq2PointFiLMTest(unittest.TestCase):
    window_size = 59

    def setUp(self):
        torch.manual_seed(42)
        self.baseline = Seq2Point(window_size=self.window_size, hidden_dim=16).eval()
        self.query = torch.randn(3, self.window_size)

    def test_encode_decode_is_elementwise_identical_to_legacy_network(self):
        with torch.no_grad():
            legacy = self.baseline.network(self.query.unsqueeze(1))
            wrapped = self.baseline(self.query)
        torch.testing.assert_close(wrapped, legacy, rtol=0, atol=0)

    def test_m0_checkpoint_selection_excludes_untrained_epoch_zero(self):
        self.assertFalse(
            is_trained_baseline_checkpoint_improvement(0, 0.1, None, 1e-4)
        )
        self.assertTrue(
            is_trained_baseline_checkpoint_improvement(1, 1.2, None, 1e-4)
        )
        self.assertTrue(
            is_trained_baseline_checkpoint_improvement(2, 1.0, 1.2, 1e-4)
        )
        self.assertFalse(
            is_trained_baseline_checkpoint_improvement(3, 1.19995, 1.2, 1e-4)
        )

    def test_zero_init_global_and_context_film_match_baseline(self):
        global_adapter = GlobalFiLMSeq2Point(self.baseline).eval()
        context_adapter = ContextFiLMSeq2Point(
            Seq2Point(window_size=self.window_size, hidden_dim=16).eval(),
            window_size=self.window_size,
            code_dim=8,
            generator_hidden_dim=12,
        ).eval()
        context_adapter.baseline.load_state_dict(self.baseline.state_dict())
        context = torch.randn(3, 4, self.window_size)
        mask = torch.ones(3, 4, dtype=torch.bool)
        with torch.no_grad():
            reference = self.baseline(self.query)
            global_output = global_adapter(self.query)
            context_output = context_adapter(self.query, context, mask)
        torch.testing.assert_close(global_output, reference, rtol=0, atol=1e-6)
        torch.testing.assert_close(context_output, reference, rtol=0, atol=1e-6)

    def test_adapter_step_preserves_frozen_baseline_hash(self):
        adapter = GlobalFiLMSeq2Point(self.baseline)
        before = module_sha256(adapter.baseline)
        optimizer = torch.optim.SGD(
            [parameter for parameter in adapter.parameters() if parameter.requires_grad], lr=0.1
        )
        adapter.train()
        loss = adapter(self.query).square().mean()
        loss.backward()
        optimizer.step()
        self.assertEqual(before, module_sha256(adapter.baseline))
        self.assertFalse(any(parameter.requires_grad for parameter in adapter.baseline.parameters()))

    def test_uniform_selector_accepts_aggregate_only_and_rejects_labels(self):
        stats = NormalisationStats(200.0, 50.0, 0.0, 1.0)
        windows, mask, _ = select_uniform_context_windows(
            aggregate_frame(), stats=stats, window_size=11, context_k=16
        )
        self.assertEqual(tuple(windows.shape), (16, 11))
        self.assertTrue(bool(mask.all()))
        with self.assertRaises(ValueError):
            select_uniform_context_windows(
                aggregate_frame(with_labels=True), stats=stats, window_size=11, context_k=16
            )

    def test_h6_is_sealed_for_normalisation_checkpoint_threshold_and_early_labels(self):
        guard = LeakageGuard(test_house="H6")
        spec = BlockSpec("H6", "B", "2020-01-01", "2020-01-02", "unused.csv")
        for purpose in (
            "normalization_aggregate",
            "normalization_appliance",
            "checkpoint_selection",
            "threshold_selection",
        ):
            with self.assertRaises(PermissionError):
                guard.check(spec, purpose, ["time", "aggregate"])
        with self.assertRaises(PermissionError):
            guard.check(spec, "final_metrics_labels", ["time", "washing_machine"])
        guard.mark_predictions_complete()
        guard.check(spec, "final_metrics_labels", ["time", "washing_machine", "status"])

    def test_wrong_mode_changes_context_but_reuses_query(self):
        adapter = ContextFiLMSeq2Point(
            self.baseline,
            window_size=self.window_size,
            code_dim=8,
            generator_hidden_dim=12,
        ).eval()
        # Make context observable while leaving the baseline and query untouched.
        torch.nn.init.normal_(adapter.generator[-1].weight, std=0.02)
        pointer = self.query.data_ptr()
        with torch.no_grad():
            correct = adapter(self.query, torch.zeros(1, 4, self.window_size))
            wrong = adapter(self.query, torch.ones(1, 4, self.window_size))
        self.assertEqual(pointer, self.query.data_ptr())
        self.assertEqual(correct.shape, wrong.shape)
        self.assertFalse(torch.equal(correct, wrong))

    def test_all_method_timestamps_must_match(self):
        timestamps = pd.date_range("2020-01-01", periods=10, freq="6s")
        fingerprint = assert_common_evaluation_timestamps(
            {"M0": timestamps, "M1": timestamps.copy(), "M2": timestamps.copy(), "M3": timestamps.copy()}
        )
        self.assertEqual(len(fingerprint), 64)
        with self.assertRaises(RuntimeError):
            assert_common_evaluation_timestamps(
                {"M0": timestamps, "M1": timestamps + pd.Timedelta(seconds=6)}
            )

    def test_fixed_seed_balanced_sampling_is_reproducible(self):
        stats = NormalisationStats(0.0, 1.0, 0.0, 1.0)
        datasets = []
        for index, house in enumerate(("H1", "H2", "H3", "H7")):
            frame = aggregate_frame(80, with_labels=True)
            frame["aggregate"] += index
            datasets.append(
                QueryWindowDataset(
                    frame,
                    house=house,
                    stats=stats,
                    window_size=11,
                    require_target=True,
                )
            )
        combined = MultiHouseQueryDataset(datasets)
        first = list(HouseholdBalancedBatchSampler(combined, batch_size=8, seed=42, epoch_size=32))
        second = list(HouseholdBalancedBatchSampler(combined, batch_size=8, seed=42, epoch_size=32))
        self.assertEqual(first, second)

    def test_fixed_seed_adapter_result_is_reproducible(self):
        def train_once():
            torch.manual_seed(42)
            baseline = Seq2Point(window_size=self.window_size, hidden_dim=16).eval()
            adapter = GlobalFiLMSeq2Point(baseline)
            generator = torch.Generator().manual_seed(42)
            query = torch.randn(4, self.window_size, generator=generator)
            target = torch.randn(4, generator=generator)
            optimizer = torch.optim.SGD(
                [parameter for parameter in adapter.parameters() if parameter.requires_grad],
                lr=0.05,
            )
            for _ in range(3):
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.mse_loss(adapter(query).reshape(-1), target)
                loss.backward()
                optimizer.step()
            adapter.eval()
            with torch.no_grad():
                return adapter(query).clone()

        torch.testing.assert_close(train_once(), train_once(), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
