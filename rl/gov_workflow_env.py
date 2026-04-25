"""
Gymnasium adapter for GovWorkflowEnv.

Key contract:
  observation_space : Box(OBS_DIM,) float32
  action_space      : Discrete(N_ACTIONS)
  action_masks()    : np.ndarray[bool, N_ACTIONS]
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from app.env import GovWorkflowEnv
from app.models import ActionModel, ActionType, ObservationModel, PriorityMode, ServiceType
from rl.action_mask import ActionMaskComputer
from rl.feature_builder import ACTION_DECODE_TABLE, N_ACTIONS, OBS_DIM, FeatureBuilder


class GovWorkflowGymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        task_id: str = "district_backlog_easy",
        seed: int = 42,
        hard_action_mask: bool = False,
        max_non_advance_streak: int = 3,
    ):
        super().__init__()
        self.task_id = task_id
        self._seed = seed
        self._task_sampler: Optional[Callable[[], str]] = None
        self._global_step_counter: Optional[list[int]] = None
        self._hard_action_mask: bool = bool(hard_action_mask)
        self._max_non_advance_streak = max(0, int(max_non_advance_streak))
        self._non_advance_streak = 0

        self._core_env = GovWorkflowEnv()
        self._fb = FeatureBuilder()
        self._amc = ActionMaskComputer()

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(OBS_DIM,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(N_ACTIONS)

        self._current_obs: Optional[ObservationModel] = None
        self._current_pm: str = "balanced"
        self._last_at: str = "advance_time"

    @property
    def core_env(self) -> GovWorkflowEnv:
        return self._core_env

    def set_hard_action_mask(self, enabled: bool) -> None:
        self._hard_action_mask = bool(enabled)

    def set_task_sampler(
        self,
        task_sampler: Optional[Callable[[], str]],
        global_step_counter: Optional[list[int]] = None,
    ) -> None:
        self._task_sampler = task_sampler
        self._global_step_counter = global_step_counter

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        if self._task_sampler is not None:
            self.task_id = self._task_sampler()

        use_seed = seed if seed is not None else self._seed
        task_opts = {"task_id": self.task_id}
        if options:
            task_opts.update(options)

        obs_model, info = self._core_env.reset(seed=use_seed, options=task_opts)
        self._current_obs = obs_model
        self._current_pm = "balanced"
        self._last_at = "advance_time"
        self._non_advance_streak = 0

        info_dict = info.model_dump() if hasattr(info, "model_dump") else info
        if not isinstance(info_dict, dict):
            try:
                info_dict = dict(info_dict)
            except (TypeError, ValueError):
                info_dict = {}

        info_dict["fairness_gap"] = self._obs_fairness_gap(obs_model)
        return self._to_array(obs_model), info_dict

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        requested_action_idx = int(action)
        action_idx = requested_action_idx

        if self._hard_action_mask and self._current_obs is not None:
            action_idx = self._sanitize_action_idx(requested_action_idx, self.action_masks())

        action_model = self._decode_action(action_idx)
        obs_model, reward, terminated, truncated, info = self._core_env.step(action_model)

        if self._global_step_counter is not None:
            self._global_step_counter[0] += 1

        self._current_obs = obs_model
        self._last_at = action_model.action_type.value
        if getattr(action_model, "priority_mode", None) is not None:
            self._current_pm = action_model.priority_mode.value
        if action_model.action_type == ActionType.ADVANCE_TIME:
            self._non_advance_streak = 0
        else:
            self._non_advance_streak += 1

        info_dict = info.model_dump() if hasattr(info, "model_dump") else info
        if not isinstance(info_dict, dict):
            try:
                info_dict = dict(info_dict)
            except (TypeError, ValueError):
                info_dict = {}

        info_dict["fairness_gap"] = self._obs_fairness_gap(obs_model)
        info_dict["requested_action_idx"] = requested_action_idx
        info_dict["executed_action_idx"] = action_idx
        info_dict["action_mask_applied"] = bool(action_idx != requested_action_idx)
        return self._to_array(obs_model), float(reward), terminated, truncated, info_dict

    def action_masks(self) -> np.ndarray:
        if self._current_obs is None:
            return np.ones(N_ACTIONS, dtype=bool)
        mask = self._amc.compute(self._current_obs, self._current_pm)
        if self._max_non_advance_streak > 0 and self._non_advance_streak >= self._max_non_advance_streak:
            forced = np.zeros(N_ACTIONS, dtype=bool)
            forced[18] = True
            return forced
        return mask

    def render(self) -> None:
        return None

    def _to_array(self, obs: ObservationModel) -> np.ndarray:
        return self._fb.build(obs, self._current_pm, self._last_at)

    def _queue_snapshot_iter(self) -> list[Any]:
        if self._current_obs is None:
            return []
        raw = getattr(self._current_obs, "queue_snapshots", [])
        if isinstance(raw, dict):
            return list(raw.values())
        if isinstance(raw, list):
            return list(raw)
        try:
            return list(raw)
        except Exception:
            return []

    def _queue_service(self, snap: Any) -> Optional[ServiceType]:
        value = getattr(snap, "service_type", None) or getattr(snap, "service", None)
        if value is None:
            return None
        if isinstance(value, ServiceType):
            return value
        try:
            return ServiceType(str(value))
        except Exception:
            return None

    def _queue_active_cases(self, snap: Any) -> int:
        return int(getattr(snap, "total_pending", getattr(snap, "active_cases", 0)) or 0)

    def _queue_urgent_cases(self, snap: Any) -> int:
        return int(getattr(snap, "urgent_pending", getattr(snap, "urgent_cases", 0)) or 0)

    def _obs_fairness_gap(self, obs: ObservationModel) -> float:
        """
        Canonical fairness signal for RL info payload.

        Current ObservationModel exposes fairness as `fairness_index`, while
        episode-level grading uses `fairness_gap` from EpisodeStateModel.
        Keep backward-compatible fallback to avoid runtime breaks.
        """
        return float(getattr(obs, "fairness_gap", getattr(obs, "fairness_index", 0.0)) or 0.0)

    def _build_action_model(self, action_type: ActionType, **kwargs: Any) -> ActionModel:
        service = kwargs.get("service")
        target_service = kwargs.get("target_service")
        officer_delta = int(kwargs.get("officer_delta", 1) or 1)
        priority_mode = kwargs.get("priority_mode")

        candidates: list[dict[str, Any]] = []

        if action_type == ActionType.ADVANCE_TIME:
            candidates.append({"action_type": action_type})

        elif action_type == ActionType.SET_PRIORITY_MODE:
            candidates.append({"action_type": action_type, "priority_mode": priority_mode})

        elif action_type == ActionType.ASSIGN_CAPACITY and service is not None:
            candidates.extend(
                [
                    {"action_type": action_type, "service": service, "officer_delta": officer_delta},
                    {"action_type": action_type, "service_target": service, "officer_delta": officer_delta},
                    {"action_type": action_type, "capacity_assignment": {service.value: officer_delta}},
                ]
            )

        elif action_type == ActionType.REQUEST_MISSING_DOCUMENTS and service is not None:
            candidates.extend(
                [
                    {"action_type": action_type, "service": service},
                    {"action_type": action_type, "service_target": service},
                ]
            )

        elif action_type == ActionType.ESCALATE_SERVICE and service is not None:
            candidates.extend(
                [
                    {"action_type": action_type, "service": service},
                    {"action_type": action_type, "service_target": service},
                    {"action_type": action_type, "escalation_target": service},
                ]
            )

        elif action_type == ActionType.REALLOCATE_OFFICERS and service is not None and target_service is not None:
            candidates.extend(
                [
                    {
                        "action_type": action_type,
                        "service": service,
                        "target_service": target_service,
                        "officer_delta": officer_delta,
                    },
                    {
                        "action_type": action_type,
                        "reallocation_delta": {
                            service.value: -officer_delta,
                            target_service.value: officer_delta,
                        },
                    },
                ]
            )

        for candidate in candidates:
            try:
                return ActionModel(**candidate)
            except Exception:
                continue

        return ActionModel(action_type=ActionType.ADVANCE_TIME)

    def _decode_action(self, action_idx: int) -> ActionModel:
        if action_idx not in ACTION_DECODE_TABLE:
            return ActionModel(action_type=ActionType.ADVANCE_TIME)

        action_type_str, service_str, priority_mode_str, delta = ACTION_DECODE_TABLE[action_idx]
        action_type = ActionType(action_type_str)

        if action_type == ActionType.SET_PRIORITY_MODE and priority_mode_str is not None:
            return self._build_action_model(
                action_type,
                priority_mode=PriorityMode(priority_mode_str),
            )

        if action_type == ActionType.ASSIGN_CAPACITY:
            if service_str == "__most_loaded__":
                target = self._find_most_loaded_service()
            elif service_str == "__most_urgent__":
                target = self._find_most_urgent_service()
            else:
                target = ServiceType(service_str) if service_str and not service_str.startswith("__") else None

            if target is None:
                return ActionModel(action_type=ActionType.ADVANCE_TIME)

            return self._build_action_model(
                action_type,
                service=target,
                officer_delta=max(int(delta or 1), 1),
            )

        if action_type == ActionType.REQUEST_MISSING_DOCUMENTS:
            target = ServiceType(service_str) if service_str and not service_str.startswith("__") else self._find_most_loaded_service()
            if target is None:
                return ActionModel(action_type=ActionType.ADVANCE_TIME)
            return self._build_action_model(action_type, service=target)

        if action_type == ActionType.ESCALATE_SERVICE:
            target = ServiceType(service_str) if service_str and not service_str.startswith("__") else self._find_most_urgent_service()
            if target is None:
                return ActionModel(action_type=ActionType.ADVANCE_TIME)
            return self._build_action_model(action_type, service=target)

        if action_type == ActionType.REALLOCATE_OFFICERS:
            source = ServiceType(service_str)
            target = self._find_reallocation_target(source)
            if target is None:
                return ActionModel(action_type=ActionType.ADVANCE_TIME)
            return self._build_action_model(
                action_type,
                service=source,
                target_service=target,
                officer_delta=1,
            )

        return ActionModel(action_type=ActionType.ADVANCE_TIME)

    def _find_most_loaded_service(self) -> Optional[ServiceType]:
        snaps = self._queue_snapshot_iter()
        if not snaps:
            return None
        best = max(snaps, key=self._queue_active_cases)
        return self._queue_service(best)

    def _find_most_urgent_service(self) -> Optional[ServiceType]:
        snaps = [snap for snap in self._queue_snapshot_iter() if self._queue_urgent_cases(snap) > 0]
        if not snaps:
            return None
        best = max(snaps, key=lambda snap: (self._queue_urgent_cases(snap), self._queue_active_cases(snap)))
        return self._queue_service(best)

    def _find_reallocation_target(self, source: ServiceType) -> Optional[ServiceType]:
        snaps = [snap for snap in self._queue_snapshot_iter() if self._queue_service(snap) != source]
        if not snaps:
            return None
        best = max(snaps, key=self._queue_active_cases)
        if self._queue_active_cases(best) <= 0:
            return None
        return self._queue_service(best)

    def _sanitize_action_idx(self, action_idx: int, masks: np.ndarray) -> int:
        if 0 <= action_idx < N_ACTIONS and bool(masks[action_idx]):
            return action_idx

        if 0 <= 18 < N_ACTIONS and bool(masks[18]):
            return 18

        valid = np.flatnonzero(masks)
        if valid.size == 0:
            return 18
        return int(valid[0])
