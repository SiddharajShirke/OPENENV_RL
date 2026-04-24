"""
main.py — Gov Workflow OpenEnv: FastAPI HTTP wrapper.

Session model
─────────────
Every POST /reset creates a new session identified by a UUID.
All subsequent calls (step, state, grade) carry that session_id in the
request body.  Sessions are kept in a thread-safe in-memory OrderedDict.
When the store reaches max_sessions capacity the oldest session is evicted
automatically (oldest-first FIFO eviction).

IMPORTANT: the in-memory store is NOT shared across multiple OS processes.
Run with workers=1 (the default from ServerSettings) to keep this correct.

Endpoint map
────────────
GET  /health                    server + session health
POST /reset                     create session, returns session_id + obs
POST /step                      advance one simulation tick
POST /state  (GET /state)       full episode state, action_history optional
POST /grade                     task-specific deterministic grader
GET  /sessions                  list active session IDs
DELETE /sessions/{id}           remove a session
POST /api/auto_step             policy selects action, then steps
POST /api/benchmark             run multiple baseline episodes
GET  /api/openenv_compliance    OpenEnv interface compliance check
GET  /docs                      Swagger UI (FastAPI auto-generated)
GET  /redoc                     ReDoc UI (FastAPI auto-generated)
"""
from __future__ import annotations

from collections import OrderedDict
import os
from pathlib import Path
import shutil
import subprocess
from threading import Lock
import time
from typing import Any, Literal
from uuid import uuid4

from fastapi import APIRouter, Body, FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.baselines import POLICIES, run_policy_episode
from app.config import env_settings, server_settings
from app.env import GovWorkflowEnv
from app.graders import grade_episode
from app.models import (
    ActionModel,
    EpisodeStateModel,
    GraderResult,
    ObservationModel,
    StepInfoModel,
)
from app.persistence import PersistenceStore
from app.simulator import LiveSimulationSession, SimulationAgentMode, run_simulation
from app.tasks import TASKS, list_tasks
from app.training_jobs import TrainingJobManager


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STORE
# ─────────────────────────────────────────────────────────────────────────────

class SessionStore:
    """
    Thread-safe in-memory session registry.

    Design decisions:
    - Uses threading.Lock — safe for Uvicorn's single-worker async+thread model.
    - Uses OrderedDict so eviction is always oldest-first in O(1) via popitem.
    - Never imports from FastAPI.  HTTP concerns (404 conversion) stay in endpoints.
    - KeyError propagates upward and is converted to 404 there.
    """

    def __init__(self, max_sessions: int | None) -> None:
        self.store: OrderedDict[str, GovWorkflowEnv] = OrderedDict()
        self.lock = Lock()
        self.max = max_sessions

    def create(
        self,
        task_id: str,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[str, ObservationModel, dict[str, Any]]:
        env = GovWorkflowEnv(task_id=task_id)
        obs, info = env.reset(seed=seed, options=options)
        session_id = str(uuid4())
        with self.lock:
            if self.max and len(self.store) >= self.max:
                self.store.popitem(last=False)   # evict oldest
            self.store[session_id] = env
        return session_id, obs, info

    def get(self, session_id: str) -> GovWorkflowEnv:
        with self.lock:
            env = self.store.get(session_id)
        if env is None:
            raise KeyError(session_id)
        return env

    def delete(self, session_id: str) -> bool:
        with self.lock:
            return self.store.pop(session_id, None) is not None

    def active_count(self) -> int:
        with self.lock:
            return len(self.store)

    def list_ids(self) -> list[str]:
        with self.lock:
            return list(self.store.keys())


class SimulationRunStore:
    def __init__(self, max_runs: int | None = None) -> None:
        self.store: OrderedDict[str, LiveSimulationSession] = OrderedDict()
        self.lock = Lock()
        self.max = max_runs

    def create(self, run: LiveSimulationSession) -> str:
        run_id = str(uuid4())
        with self.lock:
            if self.max and len(self.store) >= self.max:
                _, evicted = self.store.popitem(last=False)
                try:
                    evicted.close()
                except Exception:
                    pass
            self.store[run_id] = run
        return run_id

    def get(self, run_id: str) -> LiveSimulationSession:
        with self.lock:
            run = self.store.get(run_id)
        if run is None:
            raise KeyError(run_id)
        return run

    def delete(self, run_id: str) -> bool:
        with self.lock:
            run = self.store.pop(run_id, None)
        if run is None:
            return False
        try:
            run.close()
        except Exception:
            pass
        return True

    def list_ids(self) -> list[str]:
        with self.lock:
            return list(self.store.keys())


# ─────────────────────────────────────────────────────────────────────────────
# GLOBALS
# ─────────────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent

persistence   = PersistenceStore(repo_root=REPO_ROOT)
sessions      = SessionStore(max_sessions=env_settings.max_sessions)
model_cache: dict[tuple[str, str], Any] = {}
model_cache_lock = Lock()
training_jobs = TrainingJobManager(repo_root=REPO_ROOT, persistence=persistence)
sim_runs      = SimulationRunStore(max_runs=max(env_settings.max_sessions, 50))


# ─────────────────────────────────────────────────────────────────────────────
# DEPENDENCY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_or_404(session_id: str) -> GovWorkflowEnv:
    """Fetch a session env by ID or raise HTTP 404."""
    try:
        return sessions.get(session_id)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found. Call POST /reset to create a new session.",
        )


def get_sim_or_404(run_id: str) -> LiveSimulationSession:
    try:
        return sim_runs.get(run_id)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Simulation run '{run_id}' not found. Call POST /api/simulation/live/start to create a live run.",
        )


def resolve_policy_or_422(policy_name: str):
    policy = POLICIES.get(policy_name)
    if policy is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unknown agent/policy '{policy_name}'. Available: {sorted(POLICIES.keys())}",
        )
    return policy


def resolve_model_path_or_422(model_path: str) -> Path:
    path = Path(model_path)
    if not path.suffix:
        path = path.with_suffix(".zip")
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    if not path.exists():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Model checkpoint not found: {path}",
        )
    return path


def load_model_cached_or_503(model_path: Path, model_type: str):
    cache_key = (str(model_path), model_type)
    with model_cache_lock:
        cached = model_cache.get(cache_key)
        if cached is not None:
            return cached
    try:
        if model_type == "maskable":
            from sb3contrib import MaskablePPO
            model = MaskablePPO.load(str(model_path))
        else:
            from sb3contrib import RecurrentPPO
            model = RecurrentPPO.load(str(model_path))
    except ModuleNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RL runtime dependencies are not available. Install requirements-rl.txt.",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to load {model_type} model from {model_path}: {exc}",
        ) from exc
    with model_cache_lock:
        model_cache[cache_key] = model
    return model


def decode_action_index(action_idx: int) -> str:
    try:
        from rl.feature_builder import ACTION_DECODE_TABLE
    except ModuleNotFoundError:
        return f"action={action_idx}"
    row = ACTION_DECODE_TABLE.get(action_idx)
    if row is None:
        return f"action={action_idx}"
    action_type, service, priority_mode, delta = row
    extras = []
    if service is not None:
        extras.append(f"service={service}")
    if priority_mode is not None:
        extras.append(f"mode={priority_mode}")
    if delta is not None:
        extras.append(f"delta={delta}")
    if extras:
        return f"{action_type}[{', '.join(extras)}]"
    return action_type


# ─────────────────────────────────────────────────────────────────────────────
# API REQUEST / RESPONSE SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    active_sessions: int
    available_tasks: list[str]


class ResetRequest(BaseModel):
    task_id: str = Field(
        default=env_settings.default_task_id,
        description="Task to run. One of the three benchmark task IDs.",
    )
    seed: int | None = Field(
        default=None,
        description=(
            "RNG seed. Omit to use the task's built-in deterministic seed. "
            "Pass an explicit integer to replay the same episode."
        ),
    )
    options: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Optional overrides forwarded verbatim to env.reset(options=...). "
            "Supported key: 'task_id' to switch tasks inside an existing session."
        ),
    )


class ResetResponse(BaseModel):
    session_id: str
    observation: ObservationModel
    info: dict[str, Any]


class StepRequest(BaseModel):
    session_id: str = Field(description="Session ID returned by POST /reset.")
    action: ActionModel


class StepResponse(BaseModel):
    session_id: str
    observation: ObservationModel
    reward: float
    done: bool
    terminated: bool
    truncated: bool
    info: StepInfoModel


class StateRequest(BaseModel):
    session_id: str = Field(description="Session ID returned by POST /reset.")
    include_action_history: bool = Field(
        default=False,
        description=(
            "When False (default) the action_history list is stripped to keep payloads small. "
            "Set True to receive the full step-by-step action log."
        ),
    )


class StateResponse(BaseModel):
    session_id: str
    state: EpisodeStateModel


class GradeRequest(BaseModel):
    session_id: str = Field(description="Session ID returned by POST /reset.")


class GradeResponse(BaseModel):
    session_id: str
    score: float = Field(ge=0.0, le=1.0, description="Episode score in [0.0, 1.0].")
    grader_name: str
    metrics: dict[str, float]


class SessionListResponse(BaseModel):
    active_sessions: int
    session_ids: list[str]


class DeleteSessionResponse(BaseModel):
    deleted: str


class TaskListResponse(BaseModel):
    tasks: list[str]


class AutoStepRequest(BaseModel):
    session_id: str = Field(description="Session ID returned by POST /reset.")
    agent_policy: str = Field(
        default="backlog_clearance",
        description="Policy name from app.baselines.POLICIES.",
    )


class AutoStepResponse(BaseModel):
    session_id: str
    agent_policy: str
    action: ActionModel
    observation: ObservationModel
    reward: float
    done: bool
    terminated: bool
    truncated: bool
    info: StepInfoModel


class BenchmarkRequest(BaseModel):
    task_id: str = Field(default=env_settings.default_task_id)
    agent_policies: list[str] = Field(
        default_factory=lambda: ["urgent_first", "oldest_first", "backlog_clearance"]
    )
    runs: int = Field(default=5, ge=1, le=30)
    max_steps: int = Field(default=500, ge=1, le=2000)
    seed_base: int | None = Field(
        default=100,
        description="Base seed — each run uses seed_base + run_index.",
    )


class BenchmarkAgentRun(BaseModel):
    run_index: int
    seed: int | None
    score: float
    reward_sum: float
    completed: int
    backlog: int
    steps: int


class BenchmarkAgentSummary(BaseModel):
    agent_policy: str
    average_score: float
    min_score: float
    max_score: float
    runs: list[BenchmarkAgentRun]


class BenchmarkResponse(BaseModel):
    task_id: str
    requested_runs: int
    agent_results: list[BenchmarkAgentSummary]


class WorkflowComponentStatus(BaseModel):
    component: str
    description: str
    available: bool
    command: str | None = None
    notes: str | None = None


class WorkflowComponentsResponse(BaseModel):
    components: list[WorkflowComponentStatus]


class OpenEnvComplianceItem(BaseModel):
    key: str
    label: str
    status: Literal["pass", "fail", "unknown"]
    detail: str


class OpenEnvComplianceResponse(BaseModel):
    checked_at: float
    items: list[OpenEnvComplianceItem]
    openenv_validate_exit_code: int | None = None
    openenv_validate_stdout_tail: str | None = None
    openenv_validate_stderr_tail: str | None = None


class WorkflowRunRequest(BaseModel):
    workflow_id: Literal["baseline_openai", "inference", "phase2_eval"]
    timeout_seconds: int = Field(default=180, ge=10, le=1200)
    max_steps: int = Field(default=40, ge=1, le=500)
    episodes: int = Field(default=3, ge=1, le=20)
    model_path: str = Field(default="results/best_model/phase2_final.zip")
    model_type: Literal["maskable", "recurrent"] = Field(default="maskable")


class WorkflowRunResponse(BaseModel):
    workflow_id: str
    command: list[str]
    exit_code: int
    duration_seconds: float
    stdout: str
    stderr: str
    timed_out: bool


class RLModelInfo(BaseModel):
    label: str
    path: str
    exists: bool
    model_type: Literal["maskable", "recurrent"]


class RLModelsResponse(BaseModel):
    models: list[RLModelInfo]


class RLRunRequest(BaseModel):
    task_id: str = Field(default=env_settings.default_task_id)
    model_path: str = Field(default="results/best_model/phase2_final.zip")
    model_type: Literal["maskable", "recurrent"] = Field(default="maskable")
    max_steps: int = Field(default=80, ge=1, le=1000)
    seed: int | None = Field(default=None)


class RLRunStep(BaseModel):
    step: int
    action_index: int
    action_label: str
    reward: float
    backlog: int
    completed: int
    sla_breaches: int
    fairness_gap: float
    done: bool


class RLRunResponse(BaseModel):
    model_path: str
    model_type: Literal["maskable", "recurrent"]
    task_id: str
    seed: int
    total_steps: int
    total_reward: float
    grader_score: float
    grader_name: str
    trace: list[RLRunStep]


class RLEvaluateRequest(BaseModel):
    model_path: str = Field(default="results/best_model/phase2_final.zip")
    model_type: Literal["auto", "maskable", "recurrent"] = Field(default="auto")
    episodes: int = Field(default=3, ge=1, le=20)
    task_ids: list[str] = Field(default_factory=list)


class RLEvaluateTaskResult(BaseModel):
    task_id: str
    grader_score: float
    total_reward: float
    total_steps: int
    total_completed: int
    total_sla_breaches: int
    fairness_gap: float


class RLEvaluateResponse(BaseModel):
    model_path: str
    model_type: Literal["auto", "maskable", "recurrent"]
    episodes: int
    average_grader_score: float
    results: list[RLEvaluateTaskResult]


class SimulationRequest(BaseModel):
    task_id: str = Field(default=env_settings.default_task_id)
    agent_mode: SimulationAgentMode = Field(default=SimulationAgentMode.baseline_policy)
    max_steps: int = Field(default=80, ge=1, le=500)
    seed: int | None = Field(default=None)
    policy_name: str = Field(default="backlog_clearance")
    model_path: str | None = Field(default=None)
    model_type: Literal["maskable", "recurrent"] = Field(default="maskable")


class SimulationStep(BaseModel):
    step: int
    day: int
    action_type: str
    action_payload: dict[str, Any]
    reward: float
    done: bool
    backlog: int
    completed: int
    sla_breaches: int
    fairness_gap: float
    escalation_budget_remaining: int
    invalid_action: bool
    last_action_error: str | None = None
    queue_rows: list[dict[str, Any]]
    action_index: int | None = None
    decision_source: str | None = None
    provider: str | None = None
    model_used: str | None = None
    llm_attempts: int | None = None
    llm_error: str | None = None
    llm_key_label: str | None = None
    repair_note: str | None = None
    switch_note: str | None = None


class SimulationResponse(BaseModel):
    task_id: str
    agent_mode: SimulationAgentMode
    seed: int
    total_reward: float
    score: float
    grader_name: str
    summary: dict[str, Any]
    trace: list[SimulationStep]


class SimulationLiveStartRequest(SimulationRequest):
    pass


class SimulationLiveStartResponse(BaseModel):
    run_id: str
    task_id: str
    agent_mode: SimulationAgentMode
    seed: int
    max_steps: int
    start_log: str
    route_plan: list[str] = Field(default_factory=list)


class SimulationLiveStepRequest(BaseModel):
    run_id: str


class SimulationLiveStepResponse(BaseModel):
    run_id: str
    done: bool
    step: SimulationStep | None = None
    step_log: str | None = None
    end_log: str | None = None
    total_reward: float
    score: float | None = None
    grader_name: str | None = None
    summary: dict[str, Any] | None = None


class SimulationLiveStateResponse(BaseModel):
    run_id: str
    state: dict[str, Any]


class TrainingJobStartRequest(BaseModel):
    phase: Literal[1, 2] = Field(default=2)
    timesteps: int = Field(default=120_000, ge=10_000, le=2_000_000)
    n_envs: int = Field(default=4, ge=1, le=16)
    seed: int | None = Field(
        default=None,
        description="When omitted, a time-based seed is auto-generated.",
    )
    config_path: str | None = Field(default=None)


class TrainingJobStopResponse(BaseModel):
    stopped: bool
    job_id: str
    status: str


class TrainingJobsListResponse(BaseModel):
    jobs: list[dict[str, Any]]


class SimulationHistoryListResponse(BaseModel):
    runs: list[dict[str, Any]]


class ComparisonHistoryCreateRequest(BaseModel):
    task_id: str
    baseline_policy: str
    model_path: str
    model_type: str
    include_llm: bool = True
    runs: int
    steps: int
    episodes: int
    seed_base: int
    result: dict[str, Any]


class ComparisonHistoryCreateResponse(BaseModel):
    comparison_id: str


class ComparisonHistoryListResponse(BaseModel):
    comparisons: list[dict[str, Any]]


class HistoryClearResponse(BaseModel):
    cleared: bool
    deleted_rows: int
    scope: str


class ComparisonHistoryRepairResponse(BaseModel):
    comparison_id: str
    repaired: bool
    detail: str


# ─────────────────────────────────────────────────────────────────────────────
# APPLICATION
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Gov Workflow OpenEnv",
    summary="Government-service workflow control — OpenEnv-compatible HTTP API",
    description=(
        "A real-world OpenEnv-style environment where an AI agent reduces avoidable "
        "administrative delay in government-service workflows through queue prioritisation, "
        "missing-document handling, officer allocation, escalation control, SLA routing, "
        "and fairness management.\n\n"
        "**Quick start**\n"
        "1. `POST /reset` → get `session_id`\n"
        "2. `POST /step` with `session_id` + `action` repeatedly\n"
        "3. `POST /grade` to get the deterministic episode score\n"
        "4. `DELETE /sessions/{session_id}` to clean up"
    ),
    version="0.3.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=server_settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static UI (optional Vite build) ─────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR   = Path(__file__).resolve().parent / "web"
VITE_WEB_DIR   = WEB_DIR / "vite-dist"
LOCAL_VITE_WEB = REPO_ROOT / "frontend" / "react" / "dist"

if VITE_WEB_DIR.joinpath("index.html").exists():
    UI_INDEX_FILE = VITE_WEB_DIR / "index.html"
    UI_ASSETS_DIR = VITE_WEB_DIR / "assets"
elif LOCAL_VITE_WEB.joinpath("index.html").exists():
    UI_INDEX_FILE = LOCAL_VITE_WEB / "index.html"
    UI_ASSETS_DIR = LOCAL_VITE_WEB / "assets"
elif WEB_DIR.joinpath("index.html").exists():
    UI_INDEX_FILE = WEB_DIR / "index.html"
    UI_ASSETS_DIR = WEB_DIR
else:
    UI_INDEX_FILE = None
    UI_ASSETS_DIR = None

if UI_ASSETS_DIR is not None and UI_ASSETS_DIR.exists():
    app.mount("/ui/assets", StaticFiles(directory=str(UI_ASSETS_DIR)), name="ui-assets")


@app.get("/", include_in_schema=False)
def root_redirect() -> RedirectResponse:
    if UI_INDEX_FILE is None:
        return RedirectResponse(url="/docs", status_code=status.HTTP_307_TEMPORARY_REDIRECT)
    return RedirectResponse(url="/ui", status_code=status.HTTP_307_TEMPORARY_REDIRECT)


@app.get("/ui", include_in_schema=False)
def ui_index() -> FileResponse:
    if UI_INDEX_FILE is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="UI bundle not found. Build frontend/react with Vite first.",
        )
    return FileResponse(UI_INDEX_FILE)


# ─────────────────────────────────────────────────────────────────────────────
# CORE OpenEnv ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["meta"], summary="Server and session health")
def health() -> HealthResponse:
    """Returns server status, version, active session count, and task list."""
    return HealthResponse(
        status="ok",
        version="0.3.0",
        active_sessions=sessions.active_count(),
        available_tasks=list_tasks(),
    )


@app.post(
    "/reset",
    response_model=ResetResponse,
    status_code=status.HTTP_200_OK,
    tags=["env"],
    summary="Create a new session and return the initial observation",
)
def reset(body: ResetRequest | None = Body(default=None)) -> ResetResponse:
    """
    Creates a fresh GovWorkflowEnv episode, registers it in the session store,
    and returns a unique session_id with the initial observation.
    Use seed for reproducible episodes.
    """
    req = body or ResetRequest()
    session_id, obs, info = sessions.create(
        task_id=req.task_id,
        seed=req.seed,
        options=req.options,
    )
    return ResetResponse(session_id=session_id, observation=obs, info=info)


@app.post(
    "/step",
    response_model=StepResponse,
    tags=["env"],
    summary="Advance the simulation by one tick",
)
def step(body: StepRequest) -> StepResponse:
    """
    Applies one ActionModel to the session's environment and returns the next
    observation, reward, termination flags, and step info.
    Returns 409 Conflict if the episode has already ended.
    """
    env = get_or_404(body.session_id)
    if env.terminated or env.truncated:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Episode has already ended (terminated or truncated). Call POST /reset to start a new episode.",
        )
    obs, reward, terminated, truncated, info = env.step(body.action)
    return StepResponse(
        session_id=body.session_id,
        observation=obs,
        reward=reward,
        done=terminated or truncated,
        terminated=terminated,
        truncated=truncated,
        info=info,
    )


@app.post(
    "/state",
    response_model=StateResponse,
    tags=["env"],
    summary="Return the full internal episode state",
)
def state_post(body: StateRequest) -> StateResponse:
    """
    Returns the complete EpisodeStateModel for the given session.
    Set include_action_history=true to receive the full step-by-step log.
    Default is false to keep response payloads small during agent loops.
    """
    env = get_or_404(body.session_id)
    episode_state = env.state()
    if not body.include_action_history:
        episode_state = episode_state.model_copy(update={"action_history": None})
    return StateResponse(session_id=body.session_id, state=episode_state)


@app.get(
    "/state",
    response_model=StateResponse,
    tags=["env"],
    summary="Return the full internal episode state (GET variant)",
)
def state_get(
    session_id: str = Query(description="Session ID returned by POST /reset."),
    include_action_history: bool = Query(
        default=False,
        description="When False (default) the action_history list is stripped.",
    ),
) -> StateResponse:
    """GET variant of /state — accepts session_id as a query parameter."""
    env = get_or_404(session_id)
    episode_state = env.state()
    if not include_action_history:
        episode_state = episode_state.model_copy(update={"action_history": None})
    return StateResponse(session_id=session_id, state=episode_state)


@app.post(
    "/grade",
    response_model=GradeResponse,
    tags=["env"],
    summary="Run the deterministic task grader for the current episode",
)
def grade(body: GradeRequest) -> GradeResponse:
    """
    Runs the task-specific deterministic grader against the current episode state
    and returns a score in [0.0, 1.0] plus per-metric breakdowns.
    Can be called at any point — not only at termination.

    GraderResult fields used:
      result.score         → episode score [0.0, 1.0]
      result.grader_name   → "easy" | "medium" | "hard"
      result.metrics       → dict of named metric floats (property on GraderResult)
    """
    env = get_or_404(body.session_id)
    result: GraderResult = grade_episode(env.state())
    return GradeResponse(
        session_id=body.session_id,
        score=result.score,
        grader_name=result.grader_name,
        metrics=result.metrics,          # .metrics is a @property → dict
    )


@app.get(
    "/sessions",
    response_model=SessionListResponse,
    tags=["meta"],
    summary="List all active session IDs",
)
def list_sessions() -> SessionListResponse:
    """Returns the count and IDs of all currently active sessions."""
    return SessionListResponse(
        active_sessions=sessions.active_count(),
        session_ids=sessions.list_ids(),
    )


@app.delete(
    "/sessions/{session_id}",
    response_model=DeleteSessionResponse,
    tags=["meta"],
    summary="Delete a session and free its memory",
)
def delete_session(session_id: str) -> DeleteSessionResponse:
    """Removes the session from the store and releases its GovWorkflowEnv instance."""
    deleted = sessions.delete(session_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found.",
        )
    return DeleteSessionResponse(deleted=session_id)


# ─────────────────────────────────────────────────────────────────────────────
# /api ROUTER — frontend + extended API
# ─────────────────────────────────────────────────────────────────────────────

api = APIRouter(prefix="/api", tags=["frontend"])


@api.get("/health", response_model=HealthResponse, summary="Health — frontend alias")
def api_health() -> HealthResponse:
    return health()


@api.get("/tasks", response_model=TaskListResponse, summary="List available tasks")
def api_tasks() -> TaskListResponse:
    return TaskListResponse(tasks=list_tasks())


@api.get("/agents", response_model=list[str], summary="List baseline agent policies")
def api_agents() -> list[str]:
    return sorted(POLICIES.keys())


@api.post("/reset", response_model=ResetResponse, summary="Reset episode — frontend alias")
def api_reset(body: ResetRequest | None = Body(default=None)) -> ResetResponse:
    return reset(body)


@api.post("/step", response_model=StepResponse, summary="Step episode — frontend alias")
def api_step(body: StepRequest) -> StepResponse:
    return step(body)


@api.post("/auto_step", response_model=AutoStepResponse, summary="Compute policy action and step once")
def api_auto_step(body: AutoStepRequest) -> AutoStepResponse:
    env = get_or_404(body.session_id)
    if env.terminated or env.truncated:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Episode has already ended. Call /api/reset first.",
        )
    policy = resolve_policy_or_422(body.agent_policy)
    obs = env.build_observation()
    action = policy(obs)
    next_obs, reward, terminated, truncated, info = env.step(action)
    return AutoStepResponse(
        session_id=body.session_id,
        agent_policy=body.agent_policy,
        action=action,
        observation=next_obs,
        reward=reward,
        done=terminated or truncated,
        terminated=terminated,
        truncated=truncated,
        info=info,
    )


@api.post("/state", response_model=StateResponse, summary="State — frontend alias")
def api_state(body: StateRequest) -> StateResponse:
    return state_post(body)


@api.post("/grade", response_model=GradeResponse, summary="Grade — frontend alias")
def api_grade(body: GradeRequest) -> GradeResponse:
    return grade(body)


@api.get("/sessions", response_model=SessionListResponse, summary="List sessions — frontend alias")
def api_sessions() -> SessionListResponse:
    return list_sessions()


@api.delete("/sessions/{session_id}", response_model=DeleteSessionResponse, summary="Delete session — frontend alias")
def api_delete_session(session_id: str) -> DeleteSessionResponse:
    return delete_session(session_id)


@api.post("/benchmark", response_model=BenchmarkResponse, summary="Run multiple baseline episodes")
def api_benchmark(body: BenchmarkRequest) -> BenchmarkResponse:
    valid_tasks = set(list_tasks())
    if body.task_id not in valid_tasks:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unknown task_id '{body.task_id}'.",
        )
    if not body.agent_policies:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="agent_policies must contain at least one policy.",
        )
    agent_results = []
    for policy_name in body.agent_policies:
        resolve_policy_or_422(policy_name)
        run_rows = []
        for run_idx in range(body.runs):
            seed = None if body.seed_base is None else body.seed_base + run_idx
            result = run_policy_episode(
                task_id=body.task_id,
                policy_name=policy_name,
                seed=seed,
                max_steps=body.max_steps,
            )
            run_rows.append(BenchmarkAgentRun(
                run_index=run_idx + 1,
                seed=seed,
                score=float(result.score),
                reward_sum=float(result.reward_sum),
                completed=int(result.completed),
                backlog=int(result.backlog),
                steps=int(result.steps),
            ))
        scores = [r.score for r in run_rows]
        agent_results.append(BenchmarkAgentSummary(
            agent_policy=policy_name,
            average_score=float(sum(scores) / len(scores)),
            min_score=float(min(scores)),
            max_score=float(max(scores)),
            runs=run_rows,
        ))
    return BenchmarkResponse(
        task_id=body.task_id,
        requested_runs=body.runs,
        agent_results=agent_results,
    )


@api.get("/workflows/components", response_model=WorkflowComponentsResponse, summary="Describe visible workflow components")
def api_workflow_components() -> WorkflowComponentsResponse:
    repo_root   = REPO_ROOT
    baseline_f  = repo_root / "baseline_openai.py"
    inference_f = repo_root / "inference.py"
    phase2_model = repo_root / "results" / "best_model" / "phase2_final.zip"
    components = [
        WorkflowComponentStatus(
            component="baseline_openai.py",
            description="CLI baseline runner using OpenAI-compatible/NVIDIA endpoints.",
            available=baseline_f.exists(),
            command=r".\.venv\3.11\Scripts\python.exe baseline_openai.py --task district_backlog_easy",
            notes="Uses API keys from environment variables.",
        ),
        WorkflowComponentStatus(
            component="inference.py",
            description="Submission-style inference runner with strict START/STEP/END logging.",
            available=inference_f.exists(),
            command=r".\.venv\3.11\Scripts\python.exe inference.py",
            notes="Reads HF/OpenAI-compatible credentials from environment variables.",
        ),
        WorkflowComponentStatus(
            component="phase2_final.zip",
            description="Trained Phase 2 PPO checkpoint used for local RL evaluation/execution.",
            available=phase2_model.exists(),
            command=r".\.venv\3.11\Scripts\python.exe -m rl.evaluate --model results/best_model/phase2_final.zip --episodes 3 --model-type maskable",
        ),
        WorkflowComponentStatus(
            component="openenv-api",
            description="Standard environment API exposed through reset/step/state/grade.",
            available=True,
            command="POST /reset, POST /step, GET+POST /state, POST /grade",
        ),
    ]
    return WorkflowComponentsResponse(components=components)


@api.get("/openenv_compliance", response_model=OpenEnvComplianceResponse, summary="Check OpenEnv interface compliance")
def api_openenv_compliance(
    run_validate: bool = Query(default=False)
) -> OpenEnvComplianceResponse:
    repo_root = REPO_ROOT
    openenv_yaml = repo_root / "openenv.yaml"
    route_paths = {getattr(r, "path", "") for r in app.routes}

    def has_path(path: str) -> bool:
        return path in route_paths

    items = [
        OpenEnvComplianceItem(
            key="typed_action_model",
            label="Typed Action model (Pydantic)",
            status="pass" if issubclass(ActionModel, BaseModel) else "fail",
            detail=f"ActionModel type={ActionModel.__name__}",
        ),
        OpenEnvComplianceItem(
            key="typed_observation_model",
            label="Typed Observation model (Pydantic)",
            status="pass" if issubclass(ObservationModel, BaseModel) else "fail",
            detail=f"ObservationModel type={ObservationModel.__name__}",
        ),
        OpenEnvComplianceItem(
            key="typed_step_info_model",
            label="Typed step info model (Pydantic)",
            status="pass" if issubclass(StepInfoModel, BaseModel) else "fail",
            detail=f"StepInfoModel type={StepInfoModel.__name__}",
        ),
        OpenEnvComplianceItem(
            key="api_step_reset_state",
            label="step/reset/state API exposed",
            status="pass" if (has_path("/reset") and has_path("/step") and has_path("/state")) else "fail",
            detail="Expected endpoints: POST /reset, POST /step, GET+POST /state",
        ),
        OpenEnvComplianceItem(
            key="openenv_yaml",
            label="openenv.yaml metadata file",
            status="pass" if openenv_yaml.exists() else "fail",
            detail=str(openenv_yaml),
        ),
    ]

    validate_rc = validate_out = validate_err = None
    if run_validate:
        openenv_bin = shutil.which("openenv")
        if openenv_bin is None:
            items.append(OpenEnvComplianceItem(
                key="openenv_validate",
                label="openenv validate execution",
                status="unknown",
                detail="openenv CLI not found in runtime PATH.",
            ))
        else:
            proc = subprocess.run(
                [openenv_bin, "validate"],
                cwd=str(repo_root),
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            validate_rc   = int(proc.returncode)
            validate_out  = (proc.stdout or "")[-4000:]
            validate_err  = (proc.stderr or "")[-2000:]
            items.append(OpenEnvComplianceItem(
                key="openenv_validate",
                label="openenv validate execution",
                status="pass" if proc.returncode == 0 else "fail",
                detail=f"Exit code: {proc.returncode}",
            ))
    else:
        items.append(OpenEnvComplianceItem(
            key="openenv_validate",
            label="openenv validate execution",
            status="unknown",
            detail="Not executed in this check. Pass run_validate=true to execute.",
        ))

    return OpenEnvComplianceResponse(
        checked_at=time.time(),
        items=items,
        openenv_validate_exit_code=validate_rc,
        openenv_validate_stdout_tail=validate_out,
        openenv_validate_stderr_tail=validate_err,
    )


@api.get("/rl_models", response_model=RLModelsResponse, summary="List available trained RL model checkpoints")
def api_rl_models() -> RLModelsResponse:
    repo_root = REPO_ROOT
    phase2   = repo_root / "results" / "best_model" / "phase2_final.zip"
    phase3   = repo_root / "results" / "best_model" / "phase3_final.zip"
    best     = repo_root / "results" / "best_model" / "best_model.zip"
    return RLModelsResponse(models=[
        RLModelInfo(label="Phase 2 Final — Maskable PPO", path=str(phase2), exists=phase2.exists(), model_type="maskable"),
        RLModelInfo(label="Best Model — Maskable PPO",    path=str(best),   exists=best.exists(),   model_type="maskable"),
        RLModelInfo(label="Phase 3 Final — Recurrent PPO",path=str(phase3), exists=phase3.exists(), model_type="recurrent"),
    ])


@api.post("/rl_run", response_model=RLRunResponse, summary="Run one episode with a trained RL checkpoint")
def api_rl_run(body: RLRunRequest) -> RLRunResponse:
    if body.task_id not in set(list_tasks()):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unknown task_id '{body.task_id}'.",
        )
    model_path = resolve_model_path_or_422(body.model_path)
    model = load_model_cached_or_503(model_path, body.model_type)
    try:
        import numpy as np
        from rl.gym_wrapper import GovWorkflowGymEnv
    except ModuleNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RL runtime dependencies are not available. Install requirements-rl.txt.",
        ) from exc

    seed = body.seed if body.seed is not None else int(TASKS[body.task_id].seed)
    env = GovWorkflowGymEnv(task_id=body.task_id, seed=seed, hard_action_mask=True)
    obs, _ = env.reset(seed=seed)
    trace: list[RLRunStep] = []
    total_reward = 0.0
    done = False
    lstm_state: Any = None
    episode_start = np.array([True], dtype=bool)

    for idx in range(1, body.max_steps + 1):
        masks = env.action_masks()
        if body.model_type == "recurrent":
            action, lstm_state = model.predict(
                obs, state=lstm_state, episode_start=episode_start, deterministic=True
            )
        else:
            from sb3contrib.common.maskable.utils import get_action_masks
            action, _ = model.predict(obs, action_masks=get_action_masks(env), deterministic=True)

        action_idx = int(action.item()) if hasattr(action, "item") else action
        if not (0 <= action_idx < masks.shape[0] and bool(masks[action_idx])):
            valid = np.flatnonzero(masks)
            action_idx = int(valid[0]) if valid.size > 0 else 18

        obs, reward, terminated, truncated, info = env.step(action_idx)
        done = bool(terminated or truncated)
        total_reward += float(reward)
        core_obs = env.core_env.build_observation()
        trace.append(RLRunStep(
            step=idx,
            action_index=action_idx,
            action_label=decode_action_index(action_idx),
            reward=float(reward),
            backlog=int(core_obs.total_backlog),
            completed=int(core_obs.total_completed),
            sla_breaches=int(core_obs.total_sla_breaches),
            fairness_gap=float(core_obs.fairness_gap),
            done=done,
        ))
        if body.model_type == "recurrent":
            episode_start = np.array([done], dtype=bool)
        if done:
            break

    final_state = env.core_env.state()
    grade_result = grade_episode(final_state)
    return RLRunResponse(
        model_path=str(model_path),
        model_type=body.model_type,
        task_id=body.task_id,
        seed=seed,
        total_steps=int(final_state.total_steps),
        total_reward=float(total_reward),
        grader_score=float(grade_result.score),
        grader_name=grade_result.grader_name,
        trace=trace,
    )


@api.post("/rl_evaluate", response_model=RLEvaluateResponse, summary="Evaluate trained model across tasks")
def api_rl_evaluate(body: RLEvaluateRequest) -> RLEvaluateResponse:
    model_path = resolve_model_path_or_422(body.model_path)
    task_ids = body.task_ids or list_tasks()
    valid_tasks = set(list_tasks())
    unknown = [t for t in task_ids if t not in valid_tasks]
    if unknown:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unknown task_id values: {unknown}",
        )
    try:
        from rl.evaluate import evaluate_model
    except ModuleNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RL evaluation dependencies are not available. Install requirements-rl.txt.",
        ) from exc
    try:
        eval_rows = evaluate_model(
            model_path=str(model_path),
            task_ids=task_ids,
            n_episodes=body.episodes,
            verbose=False,
            model_type=body.model_type,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    results = [
        RLEvaluateTaskResult(
            task_id=row.task_id,
            grader_score=float(row.grader_score),
            total_reward=float(row.total_reward),
            total_steps=int(row.total_steps),
            total_completed=int(row.total_completed),
            total_sla_breaches=int(row.total_sla_breaches),
            fairness_gap=float(row.fairness_gap),
        )
        for row in eval_rows
    ]
    avg_score = float(sum(x.grader_score for x in results) / max(len(results), 1))
    return RLEvaluateResponse(
        model_path=str(model_path),
        model_type=body.model_type,
        episodes=body.episodes,
        average_grader_score=avg_score,
        results=results,
    )


@api.post("/simulation/run", response_model=SimulationResponse, summary="Run a workflow simulation")
def api_simulation_run(body: SimulationRequest) -> SimulationResponse:
    if body.task_id not in set(list_tasks()):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unknown task_id '{body.task_id}'.",
        )
    if body.agent_mode == SimulationAgentMode.baseline_policy and body.policy_name not in POLICIES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unknown policy_name '{body.policy_name}'. Available: {sorted(POLICIES.keys())}",
        )
    try:
        run = run_simulation(
            task_id=body.task_id,
            agent_mode=body.agent_mode,
            max_steps=body.max_steps,
            seed=body.seed,
            policy_name=body.policy_name,
            model_path=body.model_path,
            model_type=body.model_type,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except ModuleNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RL runtime dependencies are unavailable. Install requirements-rl.txt.",
        ) from exc

    run_id = str(uuid4())
    if persistence.enabled:
        persistence.upsert_simulation_run(
            run_id=run_id,
            task_id=run.task_id,
            agent_mode=run.agent_mode,
            status="completed",
            payload={
                "task_id": run.task_id,
                "agent_mode": run.agent_mode,
                "seed": run.seed,
                "total_reward": run.total_reward,
                "score": run.score,
                "grader_name": run.grader_name,
                "summary": run.summary,
                "trace": run.trace,
            },
        )
    return SimulationResponse(
        task_id=run.task_id,
        agent_mode=run.agent_mode,
        seed=run.seed,
        total_reward=run.total_reward,
        score=run.score,
        grader_name=run.grader_name,
        summary=run.summary,
        trace=[SimulationStep(**row) for row in run.trace],
    )


@api.post("/simulation/live/start", response_model=SimulationLiveStartResponse, summary="Start a live step-by-step simulation")
def api_simulation_live_start(body: SimulationLiveStartRequest) -> SimulationLiveStartResponse:
    if body.task_id not in set(list_tasks()):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unknown task_id '{body.task_id}'.",
        )
    if body.agent_mode == SimulationAgentMode.baseline_policy and body.policy_name not in POLICIES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unknown policy_name '{body.policy_name}'. Available: {sorted(POLICIES.keys())}",
        )
    try:
        run = LiveSimulationSession(
            task_id=body.task_id,
            agent_mode=body.agent_mode,
            max_steps=body.max_steps,
            seed=body.seed,
            policy_name=body.policy_name,
            model_path=body.model_path,
            model_type=body.model_type,
        )
    except (ValueError, ModuleNotFoundError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
            if isinstance(exc, ValueError) else status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    run_id = sim_runs.create(run)
    if persistence.enabled:
        persistence.upsert_simulation_run(
            run_id=run_id,
            task_id=run.task_id,
            agent_mode=run.agent_mode,
            status="running",
            payload={
                "task_id": run.task_id,
                "agent_mode": run.agent_mode,
                "seed": run.seed,
                "max_steps": run.max_steps,
                "summary": None,
                "trace_len": 0,
                "route_plan": list(run.llm_route),
            },
        )
    return SimulationLiveStartResponse(
        run_id=run_id,
        task_id=run.task_id,
        agent_mode=run.agent_mode,
        seed=run.seed,
        max_steps=run.max_steps,
        start_log=run.start_line,
        route_plan=list(run.llm_route),
    )


@api.post("/simulation/live/step", response_model=SimulationLiveStepResponse, summary="Execute one step for a live simulation")
def api_simulation_live_step(body: SimulationLiveStepRequest) -> SimulationLiveStepResponse:
    run = get_sim_or_404(body.run_id)
    if run.done:
        if persistence.enabled:
            persistence.upsert_simulation_run(
                run_id=body.run_id,
                task_id=run.task_id,
                agent_mode=run.agent_mode,
                status="completed",
                payload={
                    "task_id": run.task_id,
                    "agent_mode": run.agent_mode,
                    "seed": run.seed,
                    "max_steps": run.max_steps,
                    "total_reward": float(run.total_reward),
                    "score": run.score,
                    "grader_name": run.grader_name,
                    "summary": run.summary,
                    "trace": list(run.trace),
                },
            )
        return SimulationLiveStepResponse(
            run_id=body.run_id,
            done=True,
            total_reward=float(run.total_reward),
            score=run.score,
            grader_name=run.grader_name,
            summary=run.summary,
            end_log=run.end_line,
        )
    try:
        row, step_log, done = run.step_once()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Simulation step failed: {exc}",
        ) from exc

    if persistence.enabled:
        persistence.upsert_simulation_run(
            run_id=body.run_id,
            task_id=run.task_id,
            agent_mode=run.agent_mode,
            status="completed" if done else "running",
            payload={
                "task_id": run.task_id,
                "agent_mode": run.agent_mode,
                "seed": run.seed,
                "max_steps": run.max_steps,
                "total_reward": float(run.total_reward),
                "score": run.score,
                "grader_name": run.grader_name,
                "summary": run.summary,
                "trace": list(run.trace) if done else [],
                "trace_len": len(run.trace),
            },
        )
    return SimulationLiveStepResponse(
        run_id=body.run_id,
        done=done,
        step=SimulationStep(**row),
        step_log=step_log,
        end_log=run.end_line if done else None,
        total_reward=float(run.total_reward),
        score=run.score,
        grader_name=run.grader_name,
        summary=run.summary,
    )


@api.get("/simulation/live/{run_id}", response_model=SimulationLiveStateResponse, summary="Get live simulation state")
def api_simulation_live_state(run_id: str) -> SimulationLiveStateResponse:
    run = get_sim_or_404(run_id)
    return SimulationLiveStateResponse(run_id=run_id, state=run.snapshot)


@api.post("/simulation/live/{run_id}/stop", response_model=dict, summary="Stop and remove a live simulation run")
def api_simulation_live_stop(run_id: str) -> dict[str, Any]:
    run: LiveSimulationSession | None = None
    try:
        run = sim_runs.get(run_id)
    except Exception:
        run = None
    deleted = sim_runs.delete(run_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Simulation run '{run_id}' not found.",
        )
    if persistence.enabled and run is not None:
        persistence.upsert_simulation_run(
            run_id=run_id,
            task_id=run.task_id,
            agent_mode=run.agent_mode,
            status="stopped",
            payload={
                "task_id": run.task_id,
                "agent_mode": run.agent_mode,
                "seed": run.seed,
                "max_steps": run.max_steps,
                "total_reward": float(run.total_reward),
                "score": run.score,
                "grader_name": run.grader_name,
                "summary": run.summary,
                "trace_len": len(run.trace),
            },
        )
    return {"run_id": run_id, "stopped": True}


@api.get("/training_jobs", response_model=TrainingJobsListResponse, summary="List all background RL training jobs")
def api_training_jobs() -> TrainingJobsListResponse:
    return TrainingJobsListResponse(jobs=training_jobs.list_jobs())


@api.get("/training_jobs/list", response_model=TrainingJobsListResponse, summary="List training jobs — stable alias")
def api_training_jobs_list() -> TrainingJobsListResponse:
    return api_training_jobs()


@api.get("/training_jobs/{job_id}", response_model=dict, summary="Get one background RL training job")
def api_training_job(job_id: str) -> dict[str, Any]:
    job = training_jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Training job '{job_id}' not found.")
    return job


@api.post("/training_jobs", response_model=dict, summary="Start RL training in a background process")
def api_training_start(body: TrainingJobStartRequest) -> dict[str, Any]:
    try:
        import stable_baselines3  # noqa: F401
        import sb3contrib           # noqa: F401
        import gymnasium            # noqa: F401
    except ModuleNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RL training dependencies are unavailable. Install requirements-rl.txt.",
        ) from exc
    cfg = (
        body.config_path
        or ("rl/configs/curriculum.yaml" if body.phase == 2 else "rl/configs/ppo_easy.yaml")
    )
    return training_jobs.start_job(
        phase=body.phase,
        timesteps=body.timesteps,
        n_envs=body.n_envs,
        seed=body.seed,
        config_path=cfg,
    )


@api.post("/training_jobs/{job_id}/stop", response_model=TrainingJobStopResponse, summary="Stop a background training job")
def api_training_stop(job_id: str) -> TrainingJobStopResponse:
    job = training_jobs.stop_job(job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Training job '{job_id}' not found.")
    return TrainingJobStopResponse(stopped=True, job_id=job_id, status=str(job.get("status", "unknown")))


@api.delete("/training_jobs", response_model=HistoryClearResponse, summary="Clear persisted training job history")
def api_training_jobs_clear(clear_artifacts: bool = Query(default=False)) -> HistoryClearResponse:
    deleted = training_jobs.clear_jobs(clear_artifacts=clear_artifacts)
    return HistoryClearResponse(cleared=True, deleted_rows=int(deleted), scope="training_jobs")


@api.get("/history/simulations", response_model=SimulationHistoryListResponse, summary="List persisted simulation runs")
def api_history_simulations(limit: int = Query(default=20, ge=1, le=500)) -> SimulationHistoryListResponse:
    if not persistence.enabled:
        return SimulationHistoryListResponse(runs=[])
    return SimulationHistoryListResponse(runs=persistence.list_simulation_runs(limit=limit))


@api.delete("/history/simulations", response_model=HistoryClearResponse, summary="Clear persisted simulation history")
def api_history_simulations_clear() -> HistoryClearResponse:
    if not persistence.enabled:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Persistence is disabled.")
    deleted = persistence.clear_simulation_runs()
    return HistoryClearResponse(cleared=True, deleted_rows=int(deleted), scope="simulation_history")


@api.get("/history/simulations/{run_id}", response_model=dict, summary="Get one persisted simulation run")
def api_history_simulation(run_id: str) -> dict[str, Any]:
    if not persistence.enabled:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Persistence is disabled.")
    row = persistence.get_simulation_run(run_id)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Simulation history '{run_id}' not found.")
    return row


@api.post("/history/comparisons", response_model=ComparisonHistoryCreateResponse, summary="Persist a model-comparison result snapshot")
def api_history_comparison_create(body: ComparisonHistoryCreateRequest) -> ComparisonHistoryCreateResponse:
    if not persistence.enabled:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Persistence is disabled.")
    payload = body.model_dump(mode="json")
    comparison_id = persistence.create_comparison_run(payload)
    if comparison_id is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to persist comparison result.")
    return ComparisonHistoryCreateResponse(comparison_id=comparison_id)


@api.get("/history/comparisons", response_model=ComparisonHistoryListResponse, summary="List persisted model-comparison snapshots")
def api_history_comparisons(limit: int = Query(default=20, ge=1, le=500)) -> ComparisonHistoryListResponse:
    if not persistence.enabled:
        return ComparisonHistoryListResponse(comparisons=[])
    return ComparisonHistoryListResponse(comparisons=persistence.list_comparison_runs(limit=limit))


@api.get("/history/comparisons/{comparison_id}", response_model=dict, summary="Get one persisted model-comparison snapshot")
def api_history_comparison(comparison_id: str) -> dict[str, Any]:
    if not persistence.enabled:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Persistence is disabled.")
    row = persistence.get_comparison_run(comparison_id)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Comparison history '{comparison_id}' not found.")
    return row


@api.delete("/history/comparisons", response_model=HistoryClearResponse, summary="Clear persisted comparison history")
def api_history_comparisons_clear() -> HistoryClearResponse:
    if not persistence.enabled:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Persistence is disabled.")
    deleted = persistence.clear_comparison_runs()
    return HistoryClearResponse(cleared=True, deleted_rows=int(deleted), scope="comparison_history")


@api.post("/history/comparisons/{comparison_id}/repair", response_model=ComparisonHistoryRepairResponse, summary="Repair legacy comparison snapshot")
def api_history_comparison_repair(comparison_id: str) -> ComparisonHistoryRepairResponse:
    if not persistence.enabled:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Persistence is disabled.")
    row = persistence.get_comparison_run(comparison_id)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Comparison history '{comparison_id}' not found.")
    result = row.get("result") if isinstance(row.get("result"), dict) else {}
    include_llm  = bool(row.get("include_llm", True))
    has_baseline = isinstance(result.get("baselineRuns"), list) and len(result["baselineRuns"]) > 0
    has_llm      = not include_llm or (isinstance(result.get("llmRuns"), list) and len(result["llmRuns"]) > 0)
    if has_baseline and has_llm:
        return ComparisonHistoryRepairResponse(
            comparison_id=comparison_id,
            repaired=False,
            detail="No repair needed. Snapshot already contains per-run rows.",
        )
    task_id         = str(row.get("task_id") or env_settings.default_task_id)
    baseline_policy = str(row.get("baseline_policy") or "backlog_clearance")
    runs            = max(1, int(row.get("runs") or 1))
    steps           = max(1, int(row.get("steps") or 80))
    seed_base       = int(row.get("seed_base") or 100)
    baseline_runs: list[dict[str, Any]] = []
    for i in range(runs):
        seed = seed_base + i
        rr = run_policy_episode(task_id=task_id, policy_name=baseline_policy, seed=seed, max_steps=steps)
        baseline_runs.append({
            "run_index": i + 1,
            "seed": int(rr.seed),
            "score": float(rr.score),
            "reward_sum": float(rr.reward_sum),
            "completed": int(rr.completed),
            "backlog": int(rr.backlog),
        })
    llm_runs: list[dict[str, Any]] = []
    llm_error: str | None = None
    if include_llm:
        try:
            for i in range(runs):
                seed = seed_base + i
                sim = run_simulation(task_id=task_id, agent_mode=SimulationAgentMode.llm_inference,
                                     max_steps=steps, seed=seed, policy_name="backlog_clearance")
                llm_runs.append({
                    "run_index": i + 1,
                    "seed": int(sim.seed),
                    "score": float(sim.score),
                    "reward_sum": float(sim.total_reward),
                    "completed": int(sim.summary.get("total_completed", 0)),
                    "backlog": int(sim.summary.get("total_backlog", 0)),
                })
        except Exception as exc:
            llm_error = str(exc)

    baseline_score = float(sum(float(x["score"]) for x in baseline_runs) / max(1, len(baseline_runs)))
    llm_score      = float(sum(float(x["score"]) for x in llm_runs) / max(1, len(llm_runs))) if llm_runs else result.get("llmScore")
    repaired_result = dict(result)
    repaired_result["baselineScore"] = baseline_score
    repaired_result["baselineRuns"]  = baseline_runs
    repaired_result["llmRuns"]       = llm_runs
    repaired_result["llmScore"]      = llm_score
    if llm_error:
        repaired_result["llmError"]  = llm_error
    updated = dict(row)
    updated["result"]     = repaired_result
    updated["updated_at"] = time.time()
    saved_id = persistence.create_comparison_run(updated)
    if saved_id is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to persist repaired comparison snapshot.")
    return ComparisonHistoryRepairResponse(
        comparison_id=comparison_id,
        repaired=True,
        detail="Repaired legacy snapshot by backfilling per-run baseline/LLM rows.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# COMPATIBILITY ALIASES (no /api prefix — for clients that don't route through /api)
# ─────────────────────────────────────────────────────────────────────────────

app.include_router(api)

# Direct top-level aliases for all /api/* routes
for _alias, _endpoint, _method, _model in [
    ("/simulation/run",            api_simulation_run,            "POST",   SimulationResponse),
    ("/simulation/live/start",     api_simulation_live_start,     "POST",   SimulationLiveStartResponse),
    ("/simulation/live/step",      api_simulation_live_step,      "POST",   SimulationLiveStepResponse),
    ("/rl_models",                 api_rl_models,                 "GET",    RLModelsResponse),
    ("/rl_run",                    api_rl_run,                    "POST",   RLRunResponse),
    ("/rl_evaluate",               api_rl_evaluate,               "POST",   RLEvaluateResponse),
    ("/openenv_compliance",        api_openenv_compliance,        "GET",    OpenEnvComplianceResponse),
    ("/training_jobs",             api_training_jobs,             "GET",    TrainingJobsListResponse),
    ("/history/simulations",       api_history_simulations,       "GET",    SimulationHistoryListResponse),
    ("/history/comparisons",       api_history_comparisons,       "GET",    ComparisonHistoryListResponse),
]:
    if _method == "GET":
        app.get(_alias, response_model=_model, include_in_schema=False)(_endpoint)
    else:
        app.post(_alias, response_model=_model, include_in_schema=False)(_endpoint)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=server_settings.host,
        port=server_settings.port,
        log_level=server_settings.log_level,
        workers=server_settings.workers,   # always 1 for in-memory sessions
        reload=False,
    )