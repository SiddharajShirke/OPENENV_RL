"""
OpenEnv-native environment adapter for Gov Workflow.

This wraps app.env.GovWorkflowEnv without modifying the existing app runtime.
"""

from __future__ import annotations

from typing import Any, Optional

from openenv.core import Action, Environment, Observation, State

from app.env import GovWorkflowEnv
from app.models import ActionModel, EpisodeStateModel, ObservationModel


class GovWorkflowAction(Action):
    action_type: str
    service_target: Optional[str] = None
    priority_mode: Optional[str] = None
    reallocation_delta: Optional[dict[str, int]] = None
    escalation_target: Optional[str] = None
    capacity_assignment: Optional[dict[str, int]] = None
    notes: Optional[str] = None


class GovWorkflowObservation(Observation):
    observation: ObservationModel


class GovWorkflowState(State):
    state: EpisodeStateModel


class GovWorkflowOpenEnv(
    Environment[GovWorkflowAction, GovWorkflowObservation, GovWorkflowState]
):
    """OpenEnv Environment-compatible wrapper around GovWorkflowEnv."""

    def __init__(self, task_id: str = "district_backlog_easy", seed: int = 42):
        super().__init__()
        self._task_id = task_id
        self._seed = seed
        self._env = GovWorkflowEnv(task_id=task_id)
        self._last_observation: Optional[ObservationModel] = None
        self._last_reward: float | None = None
        self._last_done: bool = False

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> GovWorkflowObservation:
        del episode_id, kwargs
        effective_seed = self._seed if seed is None else int(seed)
        obs, _info = self._env.reset(seed=effective_seed)
        self._last_observation = obs
        self._last_reward = None
        self._last_done = False
        return GovWorkflowObservation(observation=obs, reward=None, done=False)

    def step(
        self,
        action: GovWorkflowAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> GovWorkflowObservation:
        del timeout_s, kwargs
        if isinstance(action, dict):
            action = GovWorkflowAction(**action)
        action_data = action.model_dump(
            exclude={"metadata"}, exclude_none=True, mode="json"
        )
        core_action = ActionModel(**action_data)
        obs, reward, terminated, truncated, _info = self._env.step(core_action)
        done = bool(terminated or truncated)
        self._last_observation = obs
        self._last_reward = float(reward)
        self._last_done = done
        return GovWorkflowObservation(
            observation=obs, reward=float(reward), done=done
        )

    @property
    def state(self) -> GovWorkflowState:
        current_state = self._env.state()
        return GovWorkflowState(
            episode_id=current_state.episode_id,
            step_count=int(current_state.total_steps),
            state=current_state,
        )

    def close(self) -> None:
        try:
            self._env.close()
        except Exception:
            pass
