"""State-conditioned Seq2Seq with the original Seq2Seq CNN/FC backbone.

The feature extractor and global hidden representation are identical to
``Seq2SeqCNN``.  The hidden vector is then split structurally into a state
branch and an amplitude branch.  State probability gates amplitude for power
prediction.  The training objective contains power and state losses (plus the
pre-existing optional house adversary); it intentionally contains no semantic
amplitude, transition, compactness, or factor-independence objectives.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_pipeline.model_registry import register_model
from model_pipeline.models.base_model import TorchNILMModel
from model_pipeline.models.seq2point.seq2point_invariant_state_aware import GradientReversal


@register_model(
    "state_conditioned_invariant_seq2seq",
    aliases=("StateConditionedInvariantSeq2Seq", "sci_seq2seq"),
    display_name="StateConditionedInvariantSeq2Seq",
)
class StateConditionedInvariantSeq2Seq(TorchNILMModel):
    display_name = "StateConditionedInvariantSeq2Seq"
    model_family = "seq2seq"
    target_type = "sequence"
    default_output_size = None

    def __init__(
        self,
        *,
        window_size: int = 599,
        hidden_dim: int = 1024,
        state_dim: int | None = None,
        amplitude_dim: int | None = None,
        state_loss_weight: float = 0.05,
        amplitude_loss_weight: float = 0.0,
        state_pos_weight: float | None = None,
        num_domains: int = 2,
        house_loss_weight: float = 0.003,
        grl_lambda: float = 1.0,
        **kwargs,
    ):
        if float(amplitude_loss_weight) != 0.0:
            raise ValueError(
                "StateConditionedInvariantSeq2Seq does not use an amplitude loss; "
                "use SemanticFactorizedSeq2Seq for semantic amplitude supervision."
            )
        if state_dim is None and amplitude_dim is None:
            state_dim = hidden_dim // 2
            amplitude_dim = hidden_dim - state_dim
        elif state_dim is None:
            state_dim = hidden_dim - int(amplitude_dim)
        elif amplitude_dim is None:
            amplitude_dim = hidden_dim - int(state_dim)

        state_dim = int(state_dim)
        amplitude_dim = int(amplitude_dim)
        if state_dim <= 0 or amplitude_dim <= 0:
            raise ValueError("state_dim and amplitude_dim must both be positive.")
        if state_dim + amplitude_dim != int(hidden_dim):
            raise ValueError("state_dim + amplitude_dim must equal hidden_dim.")

        super().__init__(
            window_size=window_size,
            hidden_dim=hidden_dim,
            state_dim=state_dim,
            amplitude_dim=amplitude_dim,
            state_loss_weight=state_loss_weight,
            amplitude_loss_weight=0.0,
            state_pos_weight=state_pos_weight,
            num_domains=num_domains,
            house_loss_weight=house_loss_weight,
            grl_lambda=grl_lambda,
            **kwargs,
        )
        self.hidden_dim = int(hidden_dim)
        self.state_dim = state_dim
        self.amplitude_dim = amplitude_dim
        self.state_loss_weight = float(state_loss_weight)
        self.amplitude_loss_weight = 0.0
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

        self.state_encoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.state_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.amplitude_encoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.amplitude_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.state_head = nn.Linear(self.state_dim, self.output_size)
        self.amplitude_head = nn.Linear(self.amplitude_dim, self.output_size)
        self.house_head = nn.Linear(self.state_dim, self.num_domains)
        self.off_level = nn.Parameter(torch.zeros(1))
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

    def encode_shared(self, x: torch.Tensor) -> torch.Tensor:
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

    def encode_state_amplitude(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        representation = self.encode_shared(x)
        return (
            self.state_encoder(representation),
            self.amplitude_encoder(representation),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z_s, _ = self.encode_state_amplitude(x)
        return z_s

    def _predict_from_state_amplitude(
        self,
        z_s: torch.Tensor,
        z_a: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state_logits = self.state_head(z_s)
        amplitude_delta = self.amplitude_head(z_a)
        state_probability = torch.sigmoid(state_logits)
        power = self.off_level + state_probability * amplitude_delta
        return power, state_logits, amplitude_delta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_s, z_a = self.encode_state_amplitude(x)
        power, _, _ = self._predict_from_state_amplitude(z_s, z_a)
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

    def _state_conditioned_house_loss(
        self,
        z_s: torch.Tensor,
        state_targets: torch.Tensor,
        house_ids: torch.Tensor,
    ) -> torch.Tensor:
        house_logits = self.house_head(
            GradientReversal.apply(z_s, self.grl_lambda)
        )
        window_states = (state_targets >= 0.5).any(dim=1)
        losses = []
        for state_value in (False, True):
            mask = window_states == state_value
            if bool(mask.any()):
                losses.append(F.cross_entropy(house_logits[mask], house_ids[mask]))
        if not losses:
            return z_s.new_zeros(())
        return torch.stack(losses).mean()

    def compute_loss(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        house_ids: torch.Tensor | None = None,
        criterion=None,
    ):
        if criterion is None:
            criterion = nn.MSELoss()

        z_s, z_a = self.encode_state_amplitude(inputs)
        outputs, state_logits, _ = self._predict_from_state_amplitude(z_s, z_a)
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
                "state_conditioned_house_loss": zero.detach(),
            }

        state_targets = state_targets.to(device=inputs.device)
        self.last_state_logits = state_logits
        state_loss = self._state_loss(state_logits, state_targets)

        if house_ids is None:
            house_loss = zero
        else:
            house_loss = self._state_conditioned_house_loss(
                z_s,
                state_targets,
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
            "state_conditioned_house_loss": house_loss.detach(),
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
