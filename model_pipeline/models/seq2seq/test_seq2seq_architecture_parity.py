from __future__ import annotations

import unittest

import torch

from model_pipeline.models.seq2seq.seq2seq import Seq2SeqCNN
from model_pipeline.models.seq2seq.seq2seq_state_amplitude_factorized import (
    StateAmplitudeFactorizedSeq2Seq,
)
from model_pipeline.models.seq2seq.seq2seq_state_aware_invariant import (
    StateAwareInvariantSeq2Seq,
)
from model_pipeline.models.seq2seq.seq2seq_state_conditioned_invariant import (
    StateConditionedInvariantSeq2Seq,
)


class Seq2SeqArchitectureParityTest(unittest.TestCase):
    def _models(self):
        common = {"window_size": 99, "output_size": 99, "hidden_dim": 32}
        return (
            Seq2SeqCNN(**common),
            StateAwareInvariantSeq2Seq(**common),
            StateConditionedInvariantSeq2Seq(**common),
            StateAmplitudeFactorizedSeq2Seq(**common),
        )

    def test_backbones_are_identical(self):
        models = self._models()
        expected_convolutions = [
            ((1, 30, 10), (2,)),
            ((30, 30, 8), (2,)),
            ((30, 40, 6), (1,)),
            ((40, 50, 5), (1,)),
            ((50, 50, 5), (1,)),
        ]
        for model in models:
            convolutions = [
                model.conv1,
                model.conv2,
                model.conv3,
                model.conv4,
                model.conv5,
            ]
            observed = [
                ((layer.in_channels, layer.out_channels, layer.kernel_size[0]), layer.stride)
                for layer in convolutions
            ]
            self.assertEqual(observed, expected_convolutions)
            self.assertEqual(model.fc1.weight.shape, (32, 300))

    def test_state_aware_state_head_does_not_gate_power(self):
        _, model, _, _ = self._models()
        model.eval()
        inputs = torch.randn(2, 99)
        power_before = model(inputs).detach()
        model.state_head.weight.data.zero_()
        model.state_head.bias.data.fill_(100.0)
        power_after = model(inputs).detach()
        self.assertTrue(torch.equal(power_before, power_after))

    def test_state_conditioned_state_head_gates_amplitude(self):
        _, _, model, _ = self._models()
        model.eval()
        inputs = torch.randn(2, 99)
        model.state_head.weight.data.zero_()
        model.amplitude_head.weight.data.zero_()
        model.amplitude_head.bias.data.fill_(2.0)
        model.off_level.data.zero_()

        model.state_head.bias.data.fill_(-100.0)
        off_power = model(inputs)
        model.state_head.bias.data.fill_(100.0)
        on_power = model(inputs)

        self.assertEqual(off_power.abs().max().item(), 0.0)
        self.assertTrue(torch.allclose(on_power, torch.full_like(on_power, 2.0)))

    def test_state_conditioned_rejects_semantic_amplitude_loss(self):
        with self.assertRaisesRegex(ValueError, "SemanticFactorizedSeq2Seq"):
            StateConditionedInvariantSeq2Seq(amplitude_loss_weight=0.1)

    def test_factorized_adapter_state_head_gates_on_amplitude(self):
        _, _, _, model = self._models()
        model.eval()
        inputs = torch.randn(2, 99)
        adapter = model.factorization_adapter
        adapter.state_head.weight.data.zero_()
        adapter.amplitude_head.weight.data.zero_()
        adapter.amplitude_head.bias.data.fill_(2.0)
        adapter.off_level.data.zero_()

        adapter.state_head.bias.data.fill_(-100.0)
        off_power = model(inputs)
        adapter.state_head.bias.data.fill_(100.0)
        on_power = model(inputs)

        self.assertEqual(off_power.abs().max().item(), 0.0)
        self.assertTrue(torch.allclose(on_power, torch.full_like(on_power, 2.0)))

    def test_factorized_amplitude_loss_only_uses_true_on_samples(self):
        _, _, _, model = self._models()
        amplitude = torch.tensor([[1.0, 100.0, 3.0]])
        power = torch.tensor([[2.0, 0.0, 5.0]])
        state = torch.tensor([[1.0, 0.0, 1.0]])
        loss = model._conditional_amplitude_loss(amplitude, power, state)
        expected = torch.nn.functional.smooth_l1_loss(
            torch.tensor([1.0, 3.0]),
            torch.tensor([2.0, 5.0]),
        )
        self.assertTrue(torch.allclose(loss, expected))

    def test_factorized_amplitude_loss_is_zero_without_on_samples(self):
        _, _, _, model = self._models()
        amplitude = torch.randn(2, 99, requires_grad=True)
        power = torch.randn(2, 99)
        state = torch.zeros(2, 99)
        loss = model._conditional_amplitude_loss(amplitude, power, state)
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        self.assertTrue(torch.equal(amplitude.grad, torch.zeros_like(amplitude)))


if __name__ == "__main__":
    unittest.main()
