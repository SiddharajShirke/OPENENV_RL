"""
Typed HTTP client for Gov Workflow OpenEnv.

This keeps a simple OpenEnv-style client interface:
    reset() -> observation wrapper
    step(action) -> step wrapper
    state() -> state wrapper
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests

from app.models import ActionModel, EpisodeStateModel, ObservationModel, StepInfoModel


@dataclass
class ClientStepResult:
    observation: ObservationModel
    reward: float
    done: bool
    terminated: bool
    truncated: bool
    info: StepInfoModel


class GovWorkflowClient:
    """Small typed client for the FastAPI deployment."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.session_id: str | None = None

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        response = requests.post(f"{self.base_url}{path}", json=body, timeout=30)
        response.raise_for_status()
        return response.json()

    def reset(self, task_id: str = "district_backlog_easy", seed: int | None = None) -> ObservationModel:
        payload: dict[str, Any] = {"task_id": task_id}
        if seed is not None:
            payload["seed"] = seed
        data = self._post("/reset", payload)
        self.session_id = data["session_id"]
        return ObservationModel(**data["observation"])

    def step(self, action: ActionModel) -> ClientStepResult:
        if not self.session_id:
            raise RuntimeError("Session not initialized. Call reset() first.")
        data = self._post(
            "/step",
            {
                "session_id": self.session_id,
                "action": action.model_dump(exclude_none=True),
            },
        )
        return ClientStepResult(
            observation=ObservationModel(**data["observation"]),
            reward=float(data["reward"]),
            done=bool(data["done"]),
            terminated=bool(data["terminated"]),
            truncated=bool(data["truncated"]),
            info=StepInfoModel(**data["info"]),
        )

    def state(self, include_action_history: bool = False) -> EpisodeStateModel:
        if not self.session_id:
            raise RuntimeError("Session not initialized. Call reset() first.")
        data = self._post(
            "/state",
            {
                "session_id": self.session_id,
                "include_action_history": include_action_history,
            },
        )
        return EpisodeStateModel(**data["state"])
