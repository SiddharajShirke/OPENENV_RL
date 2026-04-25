"""
Computes a boolean action mask (length N_ACTIONS) from the current observation.
True  = action is structurally valid right now.
False = action is impossible/wasteful; MaskablePPO will zero its logit.
"""

from __future__ import annotations

import numpy as np
from app.models import ObservationModel
from rl.feature_builder import ACTION_DECODE_TABLE, N_ACTIONS


class ActionMaskComputer:
    """
    Usage:
        amc = ActionMaskComputer()
        mask = amc.compute(obs, current_priority_mode)
    """

    def compute(
        self,
        obs: ObservationModel,
        current_priority_mode: str = "balanced",
    ) -> np.ndarray:
        mask = np.ones(N_ACTIONS, dtype=bool)

        queue_snaps = obs.queue_snapshots.values() if isinstance(obs.queue_snapshots, dict) else obs.queue_snapshots
        snapshots = {
            (snap.service_type.value if hasattr(snap.service_type, "value") else snap.service_type): snap 
            for snap in queue_snaps
        }
        active_services = {
            service for service, snap in snapshots.items()
            if getattr(snap, "active_cases", getattr(snap, "total_pending", 0)) > 0
        }
        escalation_budget = obs.escalation_budget_remaining

        services_with_missing_docs = {
            (snap.service_type.value if hasattr(snap.service_type, "value") else snap.service_type) 
            for snap in queue_snaps
            if getattr(snap, "missing_docs_cases", getattr(snap, "blocked_missing_docs", 0)) > 0
        }
        services_with_escalatable = {
            (snap.service_type.value if hasattr(snap.service_type, "value") else snap.service_type)
            for snap in queue_snaps
            if (getattr(snap, "active_cases", getattr(snap, "total_pending", 0)) - getattr(snap, "escalated_cases", getattr(snap, "urgent_pending", 0))) > 0
        }

        allocations = {}
        for service_key, value in (getattr(obs.officer_pool, "allocated", getattr(obs.officer_pool, "allocations", {})) or {}).items():
            name = service_key.value if hasattr(service_key, "value") else str(service_key)
            allocations[name] = int(value)

        idle_officers = getattr(obs.officer_pool, "idle_officers", getattr(obs.officer_pool, "reserve_officers", 0))

        for action_idx, (action_type, service, priority_mode, delta) in ACTION_DECODE_TABLE.items():

            if action_type == "set_priority_mode":
                if priority_mode == current_priority_mode:
                    mask[action_idx] = False

            elif action_type == "request_missing_documents":
                mask[action_idx] = service in services_with_missing_docs

            elif action_type == "escalate_service":
                mask[action_idx] = (
                    escalation_budget > 0
                    and service in services_with_escalatable
                )

            elif action_type == "advance_time":
                mask[action_idx] = True

            elif action_type == "reallocate_officers":
                has_source = (allocations.get(service, 0) > 0) and (service in active_services)
                has_target = any(svc != service for svc in active_services)
                mask[action_idx] = has_source and has_target

            elif action_type == "assign_capacity":
                if idle_officers <= 0:
                    mask[action_idx] = False
                elif service == "__most_loaded__":
                    mask[action_idx] = len(active_services) > 0
                elif service == "__most_urgent__":
                    mask[action_idx] = any(
                        getattr(snap, "urgent_cases", getattr(snap, "urgent_pending", 0)) > 0 for snap in queue_snaps
                    )
                else:
                    mask[action_idx] = False

        # Guarantee at least one safe action for MaskablePPO.
        if not mask.any():
            mask[18] = True

        return mask


def compute_mask(obs: ObservationModel, current_priority_mode: str = "balanced") -> np.ndarray:
    """Module-level convenience function."""
    return ActionMaskComputer().compute(obs, current_priority_mode)
