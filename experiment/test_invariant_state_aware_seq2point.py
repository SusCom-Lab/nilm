import unittest

import torch

from model_pipeline.models.seq2point.seq2point_invariant_state_aware import (
    GradientReversal,
    InvariantStateAwareSeq2Point,
)


class InvariantStateAwareSeq2PointTest(unittest.TestCase):
    def test_gradient_reversal_flips_gradient(self):
        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = GradientReversal.apply(x, 0.5).sum()
        y.backward()
        self.assertTrue(torch.equal(x.grad, torch.tensor([-0.5, -0.5, -0.5])))

    def test_model_exposes_power_state_and_house_heads(self):
        model = InvariantStateAwareSeq2Point(window_size=599, hidden_dim=16, num_domains=3)
        inputs = torch.randn(4, 599)
        targets = torch.randn(4)
        house_ids = torch.tensor([0, 1, 2, 1], dtype=torch.long)
        loss, outputs, details = model.compute_loss(inputs, targets, house_ids, torch.nn.MSELoss())

        self.assertEqual(tuple(outputs.shape), (4, 1))
        self.assertIn("power_loss", details)
        self.assertIn("state_loss", details)
        self.assertIn("house_loss", details)
        self.assertGreater(float(loss.item()), 0.0)


if __name__ == "__main__":
    unittest.main()
