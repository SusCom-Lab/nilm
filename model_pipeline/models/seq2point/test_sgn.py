from __future__ import annotations

import math
import unittest

import torch
import torch.nn as nn

from model_pipeline.model_registry import create_model
from model_pipeline.models.seq2point.sgn import (
    PAPER_FILTERS,
    PAPER_KERNEL_SIZES,
    SGN,
    SGNTower,
)


class _FixedTower(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.values = nn.Parameter(torch.tensor(values, dtype=torch.float32))

    def forward(self, inputs):
        return self.values[: len(inputs)].reshape(-1, 1)


class SGNTest(unittest.TestCase):
    def test_tower_matches_repository_architecture(self):
        tower = SGNTower(window_size=39, hidden_dim=8)
        self.assertEqual(
            tuple(layer.kernel_size[0] for layer in tower.convolutions),
            PAPER_KERNEL_SIZES,
        )
        self.assertEqual(
            tuple(layer.out_channels for layer in tower.convolutions),
            PAPER_FILTERS,
        )
        self.assertEqual(tower(torch.ones(2, 1, 39)).shape, (2, 1))

    def test_gate_operates_in_raw_power_space(self):
        model = SGN(window_size=39, hidden_dim=4)
        model.regression_tower = _FixedTower([1.0, 2.0])
        model.classification_tower = _FixedTower([0.0, math.log(1 / 3)])
        model.set_normalisation_stats({"appliance_mean": 100.0, "appliance_std": 20.0})

        gated, _, logits = model.training_outputs(torch.zeros(2, 39))
        decoded = 100.0 + 20.0 * gated
        regression = model.regression_tower(torch.zeros(2, 1, 39))
        expected = torch.sigmoid(logits) * (100.0 + 20.0 * regression)
        self.assertTrue(torch.allclose(decoded, expected))

    def test_registered_model_has_two_independent_towers(self):
        model = create_model("sgn", window_size=39, hidden_dim=8)
        self.assertIsInstance(model, SGN)
        self.assertIsNot(model.regression_tower, model.classification_tower)
        self.assertEqual(model.get_target_type(), "point")
        self.assertEqual(model.get_output_size(), 1)

    def test_forward_requires_normalisation(self):
        model = SGN(window_size=39, hidden_dim=4)
        with self.assertRaisesRegex(RuntimeError, "not configured"):
            model(torch.ones(2, 39))


if __name__ == "__main__":
    unittest.main()
