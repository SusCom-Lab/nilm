from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_pipeline.models.seq2point.seq2point_state_aware import StateAwareSeq2Point, power_to_status


class StateAwareSeq2PointMinimalTest(unittest.TestCase):
    def test_power_to_status_uses_threshold(self):
        targets = torch.tensor([0.0, 1999.0, 2000.0, 2500.0])

        status = power_to_status(targets, 2000.0)

        self.assertTrue(torch.equal(status, torch.tensor([0.0, 0.0, 1.0, 1.0])))

    def test_state_aware_seq2point_returns_power_and_state_logits(self):
        model = StateAwareSeq2Point(window_size=599, hidden_dim=32, state_threshold=0.5)
        inputs = torch.randn(4, 599)
        targets = torch.tensor([0.0, 1.0, 0.2, 2.0])

        loss, outputs = model.compute_loss(inputs, targets, criterion=torch.nn.MSELoss())

        self.assertEqual(outputs.shape, (4, 1))
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(model.last_state_logits.shape, (4,))


if __name__ == "__main__":
    unittest.main()
