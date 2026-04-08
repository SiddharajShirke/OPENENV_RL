"""
Gymnasium adapter for GovWorkflowEnv.

Key contract:
  observation_space : Box(OBS_DIM,) float32
  action_space      : Discrete(N_ACTIONS)
  action_masks()    : np.ndarray[bool, N_ACTIONS]  <- required by MaskablePPO
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Any, Callable, Optional

from app.env import GovWorkflowEnv
from app.models import ActionModel, ObservationModel, ActionType, ServiceType
from rl.feature_builder import (
    FeatureBuilder,
    OBS_DIM,
    N_ACTIONS,
    ACTION_DECODE_TABLE,
)
from rl.action_mask import ActionMaskComputer


class GovWorkflowGymEnv(gym.Env):
    """
    Gymnasium-compatible wrapper around GovWorkflowEnv.

    Parameters
    ----------
    task_id : str
        One of: district_backlog_easy | mixed_urgency_medium | cross_department_hard
    seed : int
        Fixed seed for deterministic episode generation.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        task_id: str = "district_backlog_easy",
        seed: int = 42,
        hard_action_mask: bool = False,
    ):
        super().__init__()
        self.task_id  = task_id
        self._seed    = seed
        self._task_sampler: Optional[Callable[[], str]] = None
        self._global_step_counter: Optional[list[int]] = None
        self._hard_action_mask: bool = bool(hard_action_mask)

        self._core_env = GovWorkflowEnv()
        self._fb       = FeatureBuilder()
        self._amc      = ActionMaskComputer()

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(N_ACTIONS)

        self._current_obs: Optional[ObservationModel] = None
        self._current_pm:  str = "balanced"
        self._last_at:     str = "advance_time"

    def set_hard_action_mask(self, enabled: bool) -> None:
        """
        When enabled, invalid policy actions are replaced with a valid masked action
        before being decoded and sent to the core environment.
        """
        self._hard_action_mask = bool(enabled)
    
    def set_task_sampler(
        self,
        task_sampler: Optional[Callable[[], str]],
        global_step_counter: Optional[list[int]] = None,
    ) -> None:
        self._task_sampler = task_sampler
        self._global_step_counter = global_step_counter

    # -- Gymnasium interface ---------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        if self._task_sampler is not None:
            self.task_id = self._task_sampler()

        use_seed  = seed if seed is not None else self._seed
        task_opts = {"task_id": self.task_id}
        if options:
            task_opts.update(options)

        obs_model, info   = self._core_env.reset(seed=use_seed, options=task_opts)
        self._current_obs = obs_model
        self._current_pm  = "balanced"
        self._last_at     = "advance_time"
        info_dict = info.model_dump() if hasattr(info, "model_dump") else info
        if not isinstance(info_dict, dict):
            try:
                info_dict = dict(info_dict)
            except (TypeError, ValueError):
                info_dict = {}
        
        # Inject global metrics into info for callback/eval access
        info_dict["fairness_gap"] = obs_model.fairness_gap
        return self._to_array(obs_model), info_dict

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        requested_action_idx = int(action)
        action_idx = requested_action_idx
        if self._hard_action_mask and self._current_obs is not None:
            action_idx = self._sanitize_action_idx(requested_action_idx, self.action_masks())

        action_model                 = self._decode_action(action_idx)
        obs_model, reward, terminated, truncated, info = self._core_env.step(action_model)
        if self._global_step_counter is not None:
            self._global_step_counter[0] += 1

        self._current_obs = obs_model
        self._last_at     = action_model.action_type.value
        if action_model.priority_mode is not None:
            self._current_pm = action_model.priority_mode.value

        info_dict = info.model_dump() if hasattr(info, "model_dump") else info
        if not isinstance(info_dict, dict):
            try:
                info_dict = dict(info_dict)
            except (TypeError, ValueError):
                info_dict = {}
        
        # Inject global metrics into info for callback/eval access
        info_dict["fairness_gap"] = obs_model.fairness_gap
        info_dict["requested_action_idx"] = requested_action_idx
        info_dict["executed_action_idx"] = action_idx
        info_dict["action_mask_applied"] = bool(action_idx != requested_action_idx)
        return self._to_array(obs_model), float(reward), terminated, truncated, info_dict

    def action_masks(self) -> np.ndarray:
        """Required by sb3_contrib.MaskablePPO."""
        if self._current_obs is None:
            return np.ones(N_ACTIONS, dtype=bool)
        return self._amc.compute(self._current_obs, self._current_pm)

    def render(self) -> None:
        pass

    # -- Internal helpers ------------------------------------------------------

    def _to_array(self, obs: ObservationModel) -> np.ndarray:
        return self._fb.build(obs, self._current_pm, self._last_at)

    def _decode_action(self, action_idx: int) -> ActionModel:
        if action_idx not in ACTION_DECODE_TABLE:
            return ActionModel(action_type=ActionType.ADVANCE_TIME)

        action_type_str, service_str, priority_mode_str, delta = ACTION_DECODE_TABLE[action_idx]

        kwargs: dict[str, Any] = {"action_type": ActionType(action_type_str)}

        if service_str is not None and not service_str.startswith("__"):
            kwargs["service"] = ServiceType(service_str)

        if action_type_str == "set_priority_mode" and priority_mode_str is not None:
            from app.models import PriorityMode
            kwargs["priority_mode"] = PriorityMode(priority_mode_str)

        if action_type_str == "reallocate_officers":
            source = ServiceType(service_str)
            target = self._find_reallocation_target(source)
            if target is None:
                return ActionModel(action_type=ActionType.ADVANCE_TIME)
            kwargs["service"] = source
            kwargs["target_service"] = target
            kwargs["officer_delta"] = 1

        if action_type_str == "assign_capacity":
            if service_str == "__most_loaded__":
                target = self._find_most_loaded_service()
            elif service_str == "__most_urgent__":
                target = self._find_most_urgent_service()
            else:
                target = None

            if target is None:
                return ActionModel(action_type=ActionType.ADVANCE_TIME)
            kwargs["service"] = target
            kwargs["officer_delta"] = max(int(delta or 1), 1)

        return ActionModel(**kwargs)

    def _find_most_loaded_service(self) -> Optional[ServiceType]:
        if self._current_obs is None:
            return None
        snaps = self._current_obs.queue_snapshots
        if not snaps:
            return None
        best = max(snaps, key=lambda snap: snap.active_cases)
        return best.service

    def _find_most_urgent_service(self) -> Optional[ServiceType]:
        if self._current_obs is None:
            return None
        snaps = self._current_obs.queue_snapshots
        if not snaps:
            return None
        urgent = [snap for snap in snaps if snap.urgent_cases > 0]
        if not urgent:
            return None
        best = max(urgent, key=lambda snap: (snap.urgent_cases, snap.active_cases))
        return best.service

    def _find_reallocation_target(self, source: ServiceType) -> Optional[ServiceType]:
        if self._current_obs is None:
            return None
        snaps = [snap for snap in self._current_obs.queue_snapshots if snap.service != source]
        if not snaps:
            return None
        best = max(snaps, key=lambda snap: snap.active_cases)
        if best.active_cases <= 0:
            return None
        return best.service

    def _sanitize_action_idx(self, action_idx: int, masks: np.ndarray) -> int:
        if 0 <= action_idx < N_ACTIONS and bool(masks[action_idx]):
            return action_idx

        # Prefer advance_time as safe fallback so the simulation keeps progressing.
        if 0 <= 18 < N_ACTIONS and bool(masks[18]):
            return 18

        valid = np.flatnonzero(masks)
        if valid.size == 0:
            return 18
        return int(valid[0])
