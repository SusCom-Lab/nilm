"""
AFHMM-SAC baseline adapted to joint multi-appliance disaggregation.
"""

from __future__ import annotations

from model_pipeline.model_registry import register_model
from model_pipeline.models.classical.afhmm import AFHMMBaseline


@register_model("afhmm_sac", aliases=("AFHMM-SAC",), display_name="AFHMM_SAC")
class AFHMMSACBaseline(AFHMMBaseline):
    display_name = "AFHMM_SAC"

    def _maybe_add_constraints(self, constraints, state_vectors) -> None:
        import cvxpy as cvx

        for appliance_name in self.appliance_order:
            if appliance_name not in self.signal_aggregates:
                continue
            appliance_usage = state_vectors[appliance_name] @ self.means_vector[appliance_name]
            constraints.append(cvx.sum(appliance_usage) <= self.signal_aggregates[appliance_name] * appliance_usage.shape[0])
