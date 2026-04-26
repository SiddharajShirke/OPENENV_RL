"""
Unified environment transport layer.

This module centralizes environment access so callers can use:
  - FastAPI HTTP transport
  - direct in-process transport
  - dynamic auto selection
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Literal, Protocol

from app.env import GovWorkflowEnv
from app.graders import grade_episode
from app.models import ActionModel, ObservationModel, StepInfoModel


TransportMode = Literal["auto", "http", "direct"]


class EnvGateway(Protocol):
    transport: TransportMode
    terminated: bool
    truncated: bool

    def reset(self) -> ObservationModel: ...

    def step(
        self, action: ActionModel
    ) -> tuple[ObservationModel, float, bool, bool, StepInfoModel]: ...

    def grade(self) -> tuple[float, str, dict[str, float]]: ...

    def close(self) -> None: ...


@dataclass
class DirectEnvGateway:
    task_id: str
    seed: int
    transport: TransportMode = "direct"

    def __post_init__(self) -> None:
        self._env = GovWorkflowEnv(task_id=self.task_id)
        self.terminated = False
        self.truncated = False

    def reset(self) -> ObservationModel:
        obs, _ = self._env.reset(seed=self.seed)
        self.terminated = False
        self.truncated = False
        return obs

    def step(
        self, action: ActionModel
    ) -> tuple[ObservationModel, float, bool, bool, StepInfoModel]:
        obs, reward, terminated, truncated, info = self._env.step(action)
        self.terminated = bool(terminated)
        self.truncated = bool(truncated)
        return obs, float(reward), bool(terminated), bool(truncated), info

    def grade(self) -> tuple[float, str, dict[str, float]]:
        result = grade_episode(self._env.state())
        return float(result.score), str(result.grader_name), dict(result.metrics)

    def close(self) -> None:
        close_fn = getattr(self._env, "close", None)
        if callable(close_fn):
            close_fn()


@dataclass
class HttpEnvGateway:
    task_id: str
    seed: int
    base_url: str
    api_prefix: str | None = None
    transport: TransportMode = "http"

    def __post_init__(self) -> None:
        try:
            import requests as _requests
        except ImportError as exc:
            raise ImportError("requests is required for HTTP transport.") from exc
        self._requests = _requests
        self._session_id: str | None = None
        self.terminated = False
        self.truncated = False
        self.base_url = self.base_url.rstrip("/")
        self._resolved_prefix = self._normalize_prefix(self.api_prefix)

    @staticmethod
    def _normalize_prefix(prefix: str | None) -> str:
        if prefix is None:
            return ""
        p = str(prefix).strip()
        if not p:
            return ""
        if not p.startswith("/"):
            p = "/" + p
        return p.rstrip("/")

    @staticmethod
    def _candidate_prefixes(explicit_prefix: str | None) -> list[str]:
        normalized_explicit = HttpEnvGateway._normalize_prefix(explicit_prefix)
        if normalized_explicit:
            return [normalized_explicit]

        env_prefix = HttpEnvGateway._normalize_prefix(os.getenv("OPENENV_ENV_API_PREFIX", ""))
        configured_candidates = os.getenv("OPENENV_ENV_API_PREFIX_CANDIDATES", "")

        candidates: list[str] = []
        for item in [env_prefix, *configured_candidates.split(",")]:
            normalized = HttpEnvGateway._normalize_prefix(item)
            if normalized not in candidates:
                candidates.append(normalized)

        # Ordered fallbacks: versioned API -> frontend API -> root OpenEnv API.
        for fallback in ["/api/v1", "/api", ""]:
            if fallback not in candidates:
                candidates.append(fallback)
        return candidates

    def _resolve_prefix(self) -> str:
        if self._resolved_prefix:
            return self._resolved_prefix
        for prefix in self._candidate_prefixes(self.api_prefix):
            try:
                response = self._requests.get(
                    f"{self.base_url}{prefix}/health",
                    timeout=3,
                )
                if response.ok:
                    self._resolved_prefix = prefix
                    return self._resolved_prefix
            except Exception:
                continue
        self._resolved_prefix = ""
        return self._resolved_prefix

    def _url(self, path: str) -> str:
        return f"{self.base_url}{self._resolve_prefix()}{path}"

    def _post(self, path: str, body: dict) -> dict:
        response = self._requests.post(
            self._url(path),
            json=body,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def reset(self) -> ObservationModel:
        payload = {"task_id": self.task_id, "seed": self.seed}
        data = self._post("/reset", payload)
        self._session_id = str(data["session_id"])
        self.terminated = False
        self.truncated = False
        return ObservationModel(**data["observation"])

    def step(
        self, action: ActionModel
    ) -> tuple[ObservationModel, float, bool, bool, StepInfoModel]:
        if not self._session_id:
            raise RuntimeError("Session is not initialized. Call reset() first.")
        data = self._post(
            "/step",
            {
                "session_id": self._session_id,
                "action": action.model_dump(exclude_none=True, mode="json"),
            },
        )
        obs = ObservationModel(**data["observation"])
        info = StepInfoModel(**data["info"])
        self.terminated = bool(data["terminated"])
        self.truncated = bool(data["truncated"])
        return (
            obs,
            float(data["reward"]),
            bool(data["terminated"]),
            bool(data["truncated"]),
            info,
        )

    def grade(self) -> tuple[float, str, dict[str, float]]:
        if not self._session_id:
            raise RuntimeError("Session is not initialized. Call reset() first.")
        data = self._post("/grade", {"session_id": self._session_id})
        return (
            float(data["score"]),
            str(data["grader_name"]),
            dict(data.get("metrics", {})),
        )

    def close(self) -> None:
        if not self._session_id:
            return
        try:
            self._requests.delete(self._url(f"/sessions/{self._session_id}"), timeout=10)
        except Exception:
            pass
        self._session_id = None


def _http_reachable(base_url: str) -> bool:
    try:
        import requests
        r = requests.get(f"{base_url.rstrip('/')}/health", timeout=3)
        return bool(r.ok)
    except Exception:
        return False


def create_env_gateway(
    *,
    task_id: str,
    seed: int,
    mode: TransportMode = "auto",
    base_url: str = "http://127.0.0.1:7860",
    api_prefix: str | None = None,
    enforce_fastapi: bool = False,
) -> EnvGateway:
    """
    Create environment gateway with dynamic transport selection.

    Behavior:
      - mode=http    -> always HTTP
      - mode=direct  -> always direct (unless enforce_fastapi=True)
      - mode=auto    -> HTTP if /health reachable, else direct fallback
    """
    if enforce_fastapi and mode == "direct":
        raise RuntimeError("Direct transport is disabled. Set mode to 'http' or 'auto'.")

    if mode == "http":
        return HttpEnvGateway(task_id=task_id, seed=seed, base_url=base_url, api_prefix=api_prefix)

    if mode == "direct":
        return DirectEnvGateway(task_id=task_id, seed=seed)

    if _http_reachable(base_url):
        return HttpEnvGateway(
            task_id=task_id,
            seed=seed,
            base_url=base_url,
            api_prefix=api_prefix,
            transport="auto",
        )

    if enforce_fastapi:
        raise RuntimeError(
            f"FastAPI gateway is required but unavailable at {base_url}. "
            "Start the API server or disable FORCE_FASTAPI_GATEWAY."
        )
    return DirectEnvGateway(task_id=task_id, seed=seed, transport="auto")
