"""State-aware Seq2Seq with the original Seq2Seq CNN/FC backbone.

This model is a controlled auxiliary-task counterpart of ``Seq2SeqCNN``:
both models use the same five convolutional layers, flatten the complete CNN
feature map, and form a global ``hidden_dim`` representation.  This variant
only adds separate sequence-level power and state heads.  State probabilities
do not gate power in this model.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_pipeline.model_registry import register_model
from model_pipeline.models.base_model import TorchNILMModel
from model_pipeline.models.seq2point.seq2point_invariant_state_aware import GradientReversal


@register_model(
    "state_aware_invariant_seq2seq",
    aliases=("StateAwareInvariantSeq2Seq", "sai_seq2seq"),
    display_name="StateAwareInvariantSeq2Seq",
)
class StateAwareInvariantSeq2Seq(TorchNILMModel):
    display_name = "StateAwareInvariantSeq2Seq"
    model_family = "seq2seq"
    target_type = "sequence"
    default_output_size = None

    def __init__(
        self,
        *,
        window_size: int = 599,
        hidden_dim: int = 1024,
        state_loss_weight: float = 0.05,
        state_pos_weight: float | None = None,
        num_domains: int = 2,
        house_loss_weight: float = 0.003,
        grl_lambda: float = 1.0,
        **kwargs,
    ):
        super().__init__(
            window_size=window_size,
            hidden_dim=hidden_dim,
            state_loss_weight=state_loss_weight,
            state_pos_weight=state_pos_weight,
            num_domains=num_domains,
            house_loss_weight=house_loss_weight,
            grl_lambda=grl_lambda,
            **kwargs,
        )
        self.hidden_dim = int(hidden_dim)
        self.state_loss_weight = float(state_loss_weight)
        self.state_pos_weight = (
            None if state_pos_weight is None else float(state_pos_weight)
        )
        self.num_domains = int(num_domains)
        self.house_loss_weight = float(house_loss_weight)
        self.grl_lambda = float(grl_lambda)

        length = self.window_size
        length = (length - 10) // 2 + 1
        length = (length - 8) // 2 + 1
        length = length - 6 + 1
        length = length - 5 + 1
        length = length - 5 + 1
        flattened = 50 * length

        # Identical feature extractor and global representation to Seq2SeqCNN.
        self.conv1 = nn.Conv1d(1, 30, 10, stride=2)
        self.conv2 = nn.Conv1d(30, 30, 8, stride=2)
        self.conv3 = nn.Conv1d(30, 40, 6, stride=1)
        self.conv4 = nn.Conv1d(40, 50, 5, stride=1)
        self.dropout1 = nn.Dropout(0.2)
        self.conv5 = nn.Conv1d(50, 50, 5, stride=1)
        self.dropout2 = nn.Dropout(0.2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(flattened, self.hidden_dim)
        self.dropout3 = nn.Dropout(0.2)

        self.power_head = nn.Linear(self.hidden_dim, self.output_size)
        self.state_head = nn.Linear(self.hidden_dim, self.output_size)
        self.house_head = nn.Linear(self.hidden_dim, self.num_domains)
        self.last_state_logits: torch.Tensor | None = None

    def prepare_targets(self, targets):
        if (
            isinstance(targets, torch.Tensor)
            and targets.ndim == 3
            and targets.size(-1) == 2
        ):
            return targets.float()
        return super().prepare_targets(targets)

    def _split_targets(
        self,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if targets.ndim == 3 and targets.size(-1) == 2:
            return targets[..., 0].float(), targets[..., 1].float()
        return super().prepare_targets(targets), None

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1).permute(0, 2, 1)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = self.dropout1(x)
        x = torch.relu(self.conv5(x))
        x = self.dropout2(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        return self.dropout3(x)

    def _predict_from_representation(
        self,
        representation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.power_head(representation), self.state_head(representation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        power, _ = self._predict_from_representation(self.encode(x))
        return power

    def _state_loss(
        self,
        state_logits: torch.Tensor,
        state_targets: torch.Tensor,
    ) -> torch.Tensor:
        pos_weight = None
        if self.state_pos_weight is not None:
            pos_weight = torch.as_tensor(
                self.state_pos_weight,
                dtype=state_logits.dtype,
                device=state_logits.device,
            )
        return F.binary_cross_entropy_with_logits(
            state_logits,
            state_targets,
            pos_weight=pos_weight,
        )

    def compute_loss(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        house_ids: torch.Tensor | None = None,
        criterion=None,
    ):
        if criterion is None:
            criterion = nn.MSELoss()

        representation = self.encode(inputs)
        outputs, state_logits = self._predict_from_representation(representation)
        power_targets, state_targets = self._split_targets(targets)
        power_loss = criterion(self.prepare_outputs(outputs), power_targets)

        zero = power_loss.new_zeros(())
        if state_targets is None:
            if house_ids is None:
                return power_loss, outputs
            return power_loss, outputs, {
                "power_loss": power_loss.detach(),
                "state_loss": zero.detach(),
                "house_loss": zero.detach(),
            }

        state_targets = state_targets.to(device=inputs.device)
        self.last_state_logits = state_logits
        state_loss = self._state_loss(state_logits, state_targets)

        if house_ids is None:
            house_loss = zero
        else:
            reversed_representation = GradientReversal.apply(
                representation,
                self.grl_lambda,
            )
            house_logits = self.house_head(reversed_representation)
            house_loss = F.cross_entropy(
                house_logits,
                house_ids.to(device=inputs.device).long(),
            )

        total_loss = (
            power_loss
            + self.state_loss_weight * state_loss
            + self.house_loss_weight * house_loss
        )
        if house_ids is None:
            return total_loss, outputs
        return total_loss, outputs, {
            "power_loss": power_loss.detach(),
            "state_loss": state_loss.detach(),
            "house_loss": house_loss.detach(),
        }

    def compute_selection_loss(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        criterion,
    ) -> torch.Tensor:
        outputs = self.prepare_outputs(self(inputs))
        power_targets, _ = self._split_targets(targets)
        return criterion(outputs, power_targets)
