from __future__ import annotations

# ── Path bootstrap ──────────────────────────────────────────────────────────
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── Load .env ────────────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv(dotenv_path=_ROOT / ".env", override=False)

import argparse
import json
import os
import random as _random
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

from app.env import GovWorkflowEnv
from app.models import (
    ActionModel,
    ActionType,
    ObservationModel,
    PriorityMode,
    ServiceType,
    StepInfoModel,
)
from app.tasks import get_task, list_tasks


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Model Registry & Per-Task Pools
# ══════════════════════════════════════════════════════════════════════════════

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

# ── Global 10-Model Sequential Pool (April 2026 — Verified on NVIDIA NIM) ────
#
# CHANGES FROM PREVIOUS VERSION:
#   REMOVED (invalid/unavailable IDs):
#     qwen/qwen3-next-80b-a3b-instruct     → invalid model ID
#     moonshotai/kimi-k2-instruct-0905     → not on NVIDIA NIM
#     deepseek-ai/deepseek-v3.2            → wrong ID (use deepseek-v3)
#     google/gemma-3-27b-it               → outdated (gemma-4 released)
#     mistralai/mixtral-8x22b-instruct-v0.1 → replaced by newer models
#   ADDED (verified April 2026):
#     deepseek-ai/deepseek-v4-flash        → FREE endpoint, 1M context
#     deepseek-ai/deepseek-r1             → reasoning, 685B MoE
#     nvidia/nemotron-3-super-120b-a12b   → hybrid Mamba-Transformer, 1M ctx
#     minimaxai/minimax-m2.7             → FREE endpoint, 230B
#     google/gemma-4-31b-it             → latest Gemma on NVIDIA NIM
#     qwen/qwen3.5-122b-a10b            → latest Qwen on NVIDIA NIM

GLOBAL_MODEL_POOL: list[str] = [
    "meta/llama-3.3-70b-instruct",          # 1. Primary
    "deepseek-ai/deepseek-v4-flash",         # 2. FREE endpoint — 1M context
    "deepseek-ai/deepseek-r1",              # 3. Reasoning — 685B MoE
    "nvidia/nemotron-3-super-120b-a12b",    # 4. NVIDIA native — 1M ctx
    "qwen/qwen3.5-122b-a10b",              # 5. Qwen3.5 — tool calling
    "deepseek-ai/deepseek-v3",             # 6. DeepSeek V3 — hybrid mode
    "minimaxai/minimax-m2.7",             # 7. FREE endpoint — 230B
    "google/gemma-4-31b-it",             # 8. Dense 31B — agentic workflows
    "microsoft/phi-4-mini-instruct",     # 9. Reliable small — last resort
    "meta/llama-3.1-8b-instruct",       # 10. Fastest safety fallback
]

# ── Free endpoint pool (KEY 2 — NVIDIA_API_KEY_2 fallback) ───────────────────
FREE_POOL: list[str] = [
    "deepseek-ai/deepseek-v4-flash",
    "minimaxai/minimax-m2.7",
    "microsoft/phi-4-mini-instruct",
    "meta/llama-3.1-8b-instruct",
]

# ── Fixed seeds ────────────────────────────────────────────────────────────────
TASK_SEEDS: dict[str, int] = {
    "district_backlog_easy": 11,
    "mixed_urgency_medium":  22,
    "cross_department_hard": 33,
}

LLM_TEMPERATURE = 0.2
LLM_TOP_P       = 0.7
LLM_MAX_TOKENS  = 512
MAX_LLM_STEPS   = 80

LLM_CALL_DELAY  = float(os.environ.get("LLM_CALL_DELAY", "12.0"))
LLM_CALL_JITTER = 1.0

# ── Enum fields that MUST be lowercase for Pydantic StrEnum ──────────────────
_ENUM_FIELDS = {"action_type", "priority_mode", "service", "target_service"}

# ── Canonical field names (Phase 2 update — do NOT use legacy names) ─────────
#   CORRECT                        WRONG (legacy)
#   snap.blocked_missing_docs  ←   snap.missing_docs_cases
#   snap.total_pending         ←   snap.active_cases
#   obs.fairness_gap           ←   obs.fairness_index


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Model Rotator
# ══════════════════════════════════════════════════════════════════════════════

class ModelRotator:
    def __init__(self, task_id: str) -> None:
        self._sequence: list[str] = GLOBAL_MODEL_POOL.copy()
        self._index = 0
        self._task_id = task_id
        self._rotation_log: list[dict[str, str]] = []

    @property
    def current(self) -> str:
        return self._sequence[self._index]

    @property
    def current_key_id(self) -> int:
        return 2 if self.current in FREE_POOL else 1

    @property
    def pool_exhausted(self) -> bool:
        return len(self._rotation_log) >= 50

    def rotate(self, reason: str = "error") -> str | None:
        old = self.current
        self._rotation_log.append({"from": old, "reason": reason})
        self._index = (self._index + 1) % len(self._sequence)
        new = self._sequence[self._index]
        print(
            f"\n  🔄 Model rotated: "
            f"{old.split('/')[-1]}  →  {new.split('/')[-1]}  ({reason})"
        )
        return new

    def summary(self) -> list[dict]:
        return list(self._rotation_log)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Result Dataclasses
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class StepRecord:
    step: int
    day: int
    action_type: str
    reward: float
    invalid: bool
    total_backlog: int
    total_completed: int
    model_used: str
    notes: list[str]


@dataclass
class EpisodeResult:
    task_id: str
    agent: str
    primary_model: str
    seed: int
    score: float
    grader_name: str
    total_steps: int
    total_reward: float
    total_completed: int
    total_sla_breaches: int
    total_invalid_actions: int
    final_day: int
    terminated: bool
    truncated: bool
    grader_metrics: dict[str, float]
    step_log: list[StepRecord]
    elapsed_seconds: float
    model_rotations: list[dict]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def summary(self) -> str:
        usage: dict[str, int] = {}
        for r in self.step_log:
            usage[r.model_used] = usage.get(r.model_used, 0) + 1
        usage_str = ", ".join(
            f"{m.split('/')[-1]} ({c})" for m, c in usage.items()
        )
        return (
            f"[{self.task_id}] agent={self.agent} "
            f"score={self.score:.3f} reward={self.total_reward:.2f} "
            f"completed={self.total_completed} breaches={self.total_sla_breaches} "
            f"invalid={self.total_invalid_actions} "
            f"rotations={len(self.model_rotations)} "
            f"day={self.final_day} steps={self.total_steps} "
            f"time={self.elapsed_seconds:.1f}s\n"
            f"  Model usage: {usage_str}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Direct Environment Wrapper
# ══════════════════════════════════════════════════════════════════════════════

class DirectEnvClient:
    """
    FIX: grade() now calls grade_episode(task_id, episode_state) correctly.
    Previous version called grade_episode(self.env.state()) — wrong signature.
    get_episode_state() returns EpisodeStateModel, not ObservationModel.
    """

    def __init__(self, task_id: str, seed: int) -> None:
        self.env = GovWorkflowEnv(task_id=task_id)
        self._seed = seed
        self._task_id = task_id
        self.terminated = False
        self.truncated = False

    def reset(self) -> ObservationModel:
        obs, _ = self.env.reset(seed=self._seed)
        self.terminated = False
        self.truncated = False
        return obs

    def step(
        self, action: ActionModel
    ) -> tuple[ObservationModel, float, bool, bool, StepInfoModel]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.terminated = terminated
        self.truncated = truncated
        return obs, reward, terminated, truncated, info

    def grade(self) -> tuple[float, str, dict[str, float]]:
        from app.graders import grade_episode
        episode_state = self.env.state()
        result = grade_episode(episode_state)
        return result.score, result.grader_name, result.metrics


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — HTTP Environment Wrapper
# ══════════════════════════════════════════════════════════════════════════════

class HttpEnvClient:
    def __init__(
        self, task_id: str, seed: int, base_url: str = "http://localhost:7860"
    ) -> None:
        try:
            import requests as _req
            self._req = _req
        except ImportError:
            raise ImportError("pip install requests  — required for --mode http")
        self._task_id = task_id
        self._seed = seed
        self._base_url = base_url.rstrip("/")
        self._session_id: str | None = None
        self.terminated = False
        self.truncated = False

    def _post(self, path: str, body: dict) -> dict:
        r = self._req.post(
            f"{self._base_url}{path}", json=body, timeout=30
        )
        r.raise_for_status()
        return r.json()

    def reset(self) -> ObservationModel:
        data = self._post("/reset", {"task_id": self._task_id, "seed": self._seed})
        self._session_id = data["session_id"]
        self.terminated = False
        self.truncated = False
        return ObservationModel(**data["observation"])

    def step(
        self, action: ActionModel
    ) -> tuple[ObservationModel, float, bool, bool, StepInfoModel]:
        data = self._post("/step", {
            "session_id": self._session_id,
            "action": action.model_dump(exclude_none=True),
        })
        obs  = ObservationModel(**data["observation"])
        info = StepInfoModel(**data["info"])
        self.terminated = data["terminated"]
        self.truncated  = data["truncated"]
        return obs, data["reward"], data["terminated"], data["truncated"], info

    def grade(self) -> tuple[float, str, dict[str, float]]:
        data = self._post("/grade", {"session_id": self._session_id})
        return data["score"], data["grader_name"], data["metrics"]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Heuristic Baseline Agent
# ══════════════════════════════════════════════════════════════════════════════

class HeuristicAgent:
    """
    Rule-based agent. Requires no API key.

    FIXED field names (Phase 2 canonical):
      snap.blocked_missing_docs  ← was snap.missing_docs_cases
      snap.total_pending         ← was snap.active_cases
    """

    def __init__(self) -> None:
        self._priority_set = False
        self._admin_action_day: int | None = None
        self._last_doc_request_day: int | None = None

    def reset(self) -> None:
        self._priority_set = False
        self._admin_action_day = None
        self._last_doc_request_day = None

    current_model = "heuristic"

    def rotation_summary(self) -> list[dict]:
        return []

    def update_reward(self, _: float) -> None:
        pass

    @staticmethod
    def _svc_key(service: str | ServiceType) -> str:
        return service.value if isinstance(service, ServiceType) else str(service)

    def act(self, obs: ObservationModel) -> ActionModel:
        snapshots = list(obs.queue_snapshots.values())

        # One admin action per simulated day; then always advance time.
        if self._admin_action_day == obs.day:
            return ActionModel(action_type=ActionType.ADVANCE_TIME)

        # 1. Set priority mode once
        if not self._priority_set:
            self._priority_set = True
            self._admin_action_day = obs.day
            return ActionModel(
                action_type=ActionType.SET_PRIORITY_MODE,
                priority_mode=PriorityMode.URGENT_FIRST,
            )

        # 2. Allocate any idle officer to the currently most loaded service.
        if obs.officer_pool.idle_officers > 0 and snapshots:
            most_loaded = max(snapshots, key=lambda s: s.total_pending)
            self._admin_action_day = obs.day
            return ActionModel(
                action_type=ActionType.ASSIGN_CAPACITY,
                capacity_assignment={most_loaded.service_type.value: 1},
            )

        days_left = obs.max_days - obs.day

        # 3. Reallocate one officer if load/officer ratio is clearly imbalanced.
        allocated = {
            self._svc_key(svc): int(off)
            for svc, off in obs.officer_pool.allocated.items()
        }
        if snapshots and len(allocated) >= 2:
            case_counts = {s.service_type.value: s.total_pending for s in snapshots}

            best_src: tuple[str, int] | None = None
            best_tgt: tuple[str, int] | None = None
            src_ratio = float("inf")
            tgt_ratio = -1.0

            for svc, officers in allocated.items():
                if officers <= 1:
                    continue
                ratio = case_counts.get(svc, 0) / max(officers, 1)
                if ratio < src_ratio:
                    src_ratio = ratio
                    best_src = (svc, officers)

            for svc, officers in allocated.items():
                ratio = case_counts.get(svc, 0) / max(officers, 1)
                if ratio > tgt_ratio:
                    tgt_ratio = ratio
                    best_tgt = (svc, officers)

            if best_src and best_tgt and best_src[0] != best_tgt[0] and tgt_ratio > src_ratio * 1.8:
                self._admin_action_day = obs.day
                return ActionModel(
                    action_type=ActionType.REALLOCATE_OFFICERS,
                    reallocation_delta={best_src[0]: -1, best_tgt[0]: 1},
                )

        # 4. Request missing docs conservatively to avoid repeatedly resetting
        # resolution days for already-requested cases.
        can_request_docs = (
            any(s.blocked_missing_docs > 0 for s in snapshots)
            and (
                self._last_doc_request_day is None
                or (obs.day - self._last_doc_request_day) >= 3
                or obs.pending_doc_resolutions == 0
            )
        )
        if can_request_docs:
            target_docs = max(
                snapshots,
                key=lambda s: (s.blocked_missing_docs, s.current_sla_risk, s.total_pending),
            )
            if target_docs.blocked_missing_docs > 0:
                self._admin_action_day = obs.day
                self._last_doc_request_day = obs.day
                return ActionModel(
                    action_type=ActionType.REQUEST_MISSING_DOCUMENTS,
                    service_target=target_docs.service_type,
                )

        # 5. Escalate in the final window when urgency is present.
        if obs.escalation_budget_remaining > 0:
            urgent_snaps = [s for s in snapshots if s.urgent_pending > 0]
            if urgent_snaps and days_left <= 5:
                target = max(urgent_snaps, key=lambda s: s.urgent_pending)
                self._admin_action_day = obs.day
                return ActionModel(
                    action_type=ActionType.ESCALATE_SERVICE,
                    escalation_target=target.service_type,
                )

        # 6. Default — progress simulation.
        return ActionModel(action_type=ActionType.ADVANCE_TIME)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — System Prompt
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are an expert government-office workflow manager AI.
Your job is to control a simulated government district office processing citizen
applications across multiple services.

SERVICES: passport, driving_license, gst_registration, income_certificate,
          caste_certificate, birth_certificate, land_registration

WORKFLOW STAGES (in order):
  submission → document_verification → field_verification → approval → issuance

YOUR GOAL: Maximise the episode score (0.0 to 1.0) by:
  - Completing as many applications as possible within SLA deadlines
  - Prioritising urgent cases (urgency level 3 > 2 > 1)
  - Keeping all services fairly served (no service left behind)
  - Using escalations sparingly — only when a case is about to breach SLA
  - Keeping officers productively busy (not idle)

QUEUE STATUS FIELDS EXPLAINED:
  backlog      = total_pending applications in queue
  missing_docs = blocked_missing_docs (stuck waiting for documents)
  urgent       = urgent_cases (high-urgency applications)
  breached     = breached_cases (already past SLA deadline)

AVAILABLE ACTIONS — return exactly ONE per turn as JSON:

1. Set queue processing order (do this FIRST on day 0 only):
   {"action_type": "set_priority_mode", "priority_mode": "urgent_first"}
   priority_mode options: urgent_first | oldest_first | balanced | backlog_clearance

2. Deploy a reserve officer to a service (day 0 only if reserves available):
   {"action_type": "assign_capacity", "service": "driving_license", "officer_delta": 1}

3. Unblock a stuck application with missing documents:
   {"action_type": "request_missing_documents", "service": "driving_license"}

4. Escalate one case to emergency priority (VERY LIMITED — use wisely):
   {"action_type": "escalate_service", "service": "income_certificate"}

5. Move officer between services (only when load ratio > 4x):
   {"action_type": "reallocate_officers", "service": "birth_certificate",
    "target_service": "driving_license", "officer_delta": 1}

6. Let one working day pass — THE ONLY ACTION THAT PROCESSES APPLICATIONS:
   {"action_type": "advance_time"}

CRITICAL RULES:
  - ALL values MUST be lowercase: driving_license NOT DRIVING_LICENSE
  - advance_time is the ONLY action that earns progress reward
  - Do NOT chain more than 2 admin actions before calling advance_time
  - Do NOT escalate before (max_days - 5) unless case already breached SLA
  - Do NOT reallocate if source service has fewer than 2 officers

OPTIMAL STRATEGY:
  Day 0:     set_priority_mode → assign_capacity (if reserves > 0) → advance_time
  Every day: request_missing_documents (ONE service, highest missing_docs) → advance_time
  Final 5:   escalate_service (urgent/breached only) → advance_time

RESPONSE FORMAT — return ONLY a raw JSON object, nothing else:
  CORRECT:   {"action_type": "advance_time"}
  CORRECT:   {"action_type": "request_missing_documents", "service": "driving_license"}
  WRONG:     ```json\n{"action_type": "ADVANCE_TIME"}```
"""


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — JSON Extraction with Lowercase Normaliser
# ══════════════════════════════════════════════════════════════════════════════

def _extract_json_action(raw: str) -> dict[str, Any]:
    cleaned = re.sub(r"```(?:json)?", "", raw).strip()
    parsed: dict[str, Any] | None = None

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    if parsed is None:
        match = re.search(r"\{[^{}]*\}", cleaned, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
            except json.JSONDecodeError:
                pass

    if parsed is None:
        print(f"  ⚠ JSON parse failed, falling back to advance_time. Raw: {raw[:120]!r}")
        return {"action_type": "advance_time"}

    for enum_field in _ENUM_FIELDS:
        if enum_field in parsed and isinstance(parsed[enum_field], str):
            parsed[enum_field] = parsed[enum_field].lower()

    return parsed


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — Observation → User Message Builder
# ══════════════════════════════════════════════════════════════════════════════

def _build_user_message(
    obs: ObservationModel, step_num: int, cumulative_reward: float
) -> str:
    """
    FIXED field names (Phase 2 canonical):
      snap.total_pending        ← was snap.active_cases
      snap.blocked_missing_docs ← was snap.missing_docs_cases
    """
    queue_lines = []
    for snap in obs.queue_snapshots:
        officers = obs.officer_pool.allocations.get(snap.service, 0)
        queue_lines.append(
            f"  {snap.service:<22}: "
            f"backlog={snap.total_pending:>3} "
            f"officers={officers} "
            f"missing_docs={snap.blocked_missing_docs:>2} "
            f"urgent={snap.urgent_cases} "
            f"breached={snap.breached_cases} "
            f"avg_age={snap.avg_age_days:.1f}d"
        )
    return (
        f"STEP {step_num} | Day {obs.day}/{obs.max_days} "
        f"| Days remaining: {obs.max_days - obs.day}\n"
        f"Cumulative reward: {cumulative_reward:.2f}\n"
        f"Priority mode: {obs.priority_mode}\n"
        f"Reserve officers: {obs.officer_pool.reserve_officers}\n"
        f"Escalation budget remaining: {obs.escalation_budget_remaining}\n"
        f"Total pending: {obs.total_backlog} "
        f"| Completed: {obs.total_completed} "
        f"| SLA breaches: {obs.total_sla_breaches}\n"
        f"Fairness gap: {obs.fairness_gap:.3f}\n\n"
        f"QUEUE STATUS:\n" + "\n".join(queue_lines) + "\n\n"
        f"Return a single JSON action object. All values lowercase."
    )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — LLM Agent with Model Rotation
# ══════════════════════════════════════════════════════════════════════════════

class LLMAgent:
    def __init__(
        self,
        task_id: str,
        model_override: str | None = None,
        api_key: str | None = None,
    ) -> None:
        try:
            from openai import OpenAI
            self._OpenAI = OpenAI
        except ImportError:
            raise ImportError("pip install openai  — required for LLM agent")

        resolved_key = api_key or os.environ.get("NVIDIA_API_KEY", "")
        self._api_key_2 = os.environ.get("NVIDIA_API_KEY_2", "")

        if not resolved_key:
            raise ValueError(
                "NVIDIA_API_KEY not set.\n"
                "  .env file : NVIDIA_API_KEY=nvapi-xxxxxxxxxxxx\n"
                "  Get free key: https://build.nvidia.com/explore/discover"
            )

        self._api_key = resolved_key
        self._task_id = task_id
        self._rotator = ModelRotator(task_id)

        if model_override:
            seq = [model_override] + [
                m for m in self._rotator._sequence if m != model_override
            ]
            self._rotator._sequence = seq

        self._client = self._OpenAI(base_url=NVIDIA_BASE_URL, api_key=self._api_key)
        self._client_2 = (
            self._OpenAI(base_url=NVIDIA_BASE_URL, api_key=self._api_key_2)
            if self._api_key_2 else None
        )
        self._history: list[dict[str, str]] = []
        self._cumulative_reward = 0.0

    @property
    def current_model(self) -> str:
        return self._rotator.current

    def reset(self) -> None:
        self._history = []
        self._cumulative_reward = 0.0
        self._rotator = ModelRotator(self._task_id)

    def update_reward(self, reward: float) -> None:
        self._cumulative_reward += reward

    def rotation_summary(self) -> list[dict]:
        return self._rotator.summary()

    def act(self, obs: ObservationModel, step_num: int) -> ActionModel:
        if self._rotator.pool_exhausted:
            print("  ⚠ Pool exhausted — returning advance_time")
            return ActionModel(action_type=ActionType.ADVANCE_TIME)

        user_message = _build_user_message(obs, step_num, self._cumulative_reward)
        self._history.append({"role": "user", "content": user_message})

        if len(self._history) > 20:
            self._history = self._history[-20:]

        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self._history
        raw_reply = ""

        while True:
            try:
                active_client = self._client
                if self._rotator.current_key_id == 2 and self._client_2:
                    active_client = self._client_2

                response = active_client.chat.completions.create(
                    model=self._rotator.current,
                    messages=messages,
                    temperature=LLM_TEMPERATURE,
                    top_p=LLM_TOP_P,
                    max_tokens=LLM_MAX_TOKENS,
                    timeout=30,
                )
                raw_reply = response.choices.message.content or ""
                break

            except KeyboardInterrupt:
                raise

            except Exception as exc:
                err_name = type(exc).__name__
                err_msg  = str(exc)[:120]
                print(f"  ⚠ {err_name} on {self._rotator.current.split('/')[-1]}: {err_msg}")
                self._rotator.rotate(reason=err_name)
                time.sleep(1.0)
                if self._rotator.pool_exhausted:
                    return ActionModel(action_type=ActionType.ADVANCE_TIME)

        self._history.append({"role": "assistant", "content": raw_reply})
        action_dict = _extract_json_action(raw_reply)

        try:
            return ActionModel(**action_dict)
        except Exception as exc:
            print(f"  ⚠ ActionModel parse failed ({exc}), using advance_time")
            return ActionModel(action_type=ActionType.ADVANCE_TIME)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — Episode Runner
# ══════════════════════════════════════════════════════════════════════════════

def run_episode(
    task_id: str,
    agent_type: str,
    model_override: str | None,
    mode: str,
    server_url: str,
    api_key: str | None,
    verbose: bool,
    max_steps: int = MAX_LLM_STEPS,
    delay_override: float | None = None,
) -> EpisodeResult:
    seed  = TASK_SEEDS.get(task_id, get_task(task_id).seed)
    delay = delay_override if delay_override is not None else LLM_CALL_DELAY

    if mode == "http":
        client: DirectEnvClient | HttpEnvClient = HttpEnvClient(
            task_id, seed, base_url=server_url
        )
    else:
        client = DirectEnvClient(task_id, seed)

    if agent_type == "llm":
        agent: HeuristicAgent | LLMAgent = LLMAgent(
            task_id=task_id,
            model_override=model_override,
            api_key=api_key,
        )
        primary_label = agent.current_model
    else:
        agent = HeuristicAgent()
        primary_label = "heuristic"

    agent.reset()
    obs = client.reset()

    step_log: list[StepRecord] = []
    total_reward = 0.0
    total_invalid = 0
    step_num = 0
    start = time.perf_counter()

    print(f"\n{'═'*65}")
    print(f"  Task  : {task_id}")
    if agent_type == "llm":
        k1 = "✅ loaded" if os.environ.get("NVIDIA_API_KEY", "") else "❌ MISSING"
        k2 = "✅ loaded" if os.environ.get("NVIDIA_API_KEY_2", "") else "⚠ not set"
        print(f"  KEY 1 : {k1}   KEY 2 : {k2}")
        pool_short = " → ".join(m.split("/")[-1][:14] for m in GLOBAL_MODEL_POOL)
        print(f"  Pool  : {pool_short}")
    print(f"  Agent : {agent_type}  |  Mode: {mode}  |  Seed: {seed}")
    print(f"  Max steps: {max_steps}  |  Delay: {delay}s")
    print(f"{'═'*65}")

    while not (client.terminated or client.truncated) and step_num < max_steps:
        step_num += 1
        current_model = agent.current_model

        if agent_type == "llm":
            action = agent.act(obs, step_num)
        else:
            action = agent.act(obs)

        obs, reward, terminated, truncated, info = client.step(action)
        agent.update_reward(reward)

        total_reward += reward
        if info.invalid_action:
            total_invalid += 1

        step_notes: list[str] = []
        legacy_notes = getattr(info, "notes", None)
        if isinstance(legacy_notes, list):
            step_notes.extend(str(n).strip() for n in legacy_notes if str(n).strip())
        elif isinstance(legacy_notes, str) and legacy_notes.strip():
            step_notes.append(legacy_notes.strip())

        if info.action_explanation.strip():
            step_notes.append(info.action_explanation.strip())
        step_notes.extend(s.strip() for s in info.effects_resolved_this_step if s.strip())
        step_notes = list(dict.fromkeys(step_notes))

        record = StepRecord(
            step=step_num,
            day=obs.day,
            action_type=action.action_type.value,
            reward=round(reward, 4),
            invalid=info.invalid_action,
            total_backlog=obs.total_backlog,
            total_completed=obs.total_completed,
            model_used=current_model,
            notes=step_notes,
        )
        step_log.append(record)

        if verbose:
            status    = "❌" if info.invalid_action else "✅"
            model_tag = (
                f"[{current_model.split('/')[-1][:22]}]"
                if agent_type == "llm" else ""
            )
            print(
                f"  step={step_num:3d} day={obs.day:2d} "
                f"action={action.action_type.value:<28} "
                f"reward={reward:+.3f}  {status}  {model_tag}"
            )
            if step_notes:
                print(f"         notes: {step_notes}")

        if agent_type == "llm":
            actual_delay = delay + _random.uniform(-LLM_CALL_JITTER, LLM_CALL_JITTER)
            if not verbose:
                print(
                    f"  Step {step_num}/{max_steps} — sleeping {actual_delay:.1f}s "
                    f"[{current_model.split('/')[-1][:20]}]",
                    end="\r", flush=True,
                )
            time.sleep(max(1.0, actual_delay))
            if not verbose:
                print(" " * 80, end="\r", flush=True)

    score, grader_name, grader_metrics = client.grade()
    elapsed = round(time.perf_counter() - start, 2)
    rotations = agent.rotation_summary()

    print(f"\n{'-'*65}")
    print(f"  SCORE  : {score:.3f} / 1.000  (grader: {grader_name})")
    print(f"  Reward : {total_reward:.2f}  |  Steps: {step_num}")
    print(f"  Completed: {obs.total_completed}  |  SLA breaches: {obs.total_sla_breaches}")
    print(f"  Invalid actions: {total_invalid}  |  Model rotations: {len(rotations)}")
    print(f"  Time: {elapsed}s")
    print(f"  Grader metrics:")
    for metric, value in grader_metrics.items():
        bar = "█" * int(value * 20)
        print(f"    {metric:<34} {value:.3f}  {bar}")
    if rotations:
        print(f"  Rotation log:")
        for r in rotations:
            print(f"    {r['from'].split('/')[-1]:<30} → rotated ({r['reason']})")
    print(f"{'-'*65}")

    return EpisodeResult(
        task_id=task_id,
        agent=agent_type,
        primary_model=primary_label,
        seed=seed,
        score=score,
        grader_name=grader_name,
        total_steps=step_num,
        total_reward=round(total_reward, 4),
        total_completed=obs.total_completed,
        total_sla_breaches=obs.total_sla_breaches,
        total_invalid_actions=total_invalid,
        final_day=obs.day,
        terminated=client.terminated,
        truncated=client.truncated,
        grader_metrics=grader_metrics,
        step_log=step_log,
        elapsed_seconds=elapsed,
        model_rotations=rotations,
    )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 12 — Reporter
# ══════════════════════════════════════════════════════════════════════════════

def save_results(results: list[EpisodeResult], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"baseline_run_{ts}.json"
    payload = {
        "run_timestamp": datetime.now().isoformat(),
        "total_episodes": len(results),
        "average_score": round(sum(r.score for r in results) / len(results), 4),
        "model_pool": GLOBAL_MODEL_POOL,
        "free_pool": FREE_POOL,
        "episodes": [asdict(r) for r in results],
    }
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


def print_leaderboard(results: list[EpisodeResult]) -> None:
    print(f"\n{'═'*72}")
    print("  LEADERBOARD")
    print(f"{'═'*72}")
    header = (
        f"  {'TASK':<32} {'MODEL':<24} {'SCORE':>7}  "
        f"{'REWARD':>8}  {'DONE':>5}  {'ROT':>4}"
    )
    print(header)
    print(f"  {'-'*32} {'-'*24} {'-'*7}  {'-'*8}  {'-'*5}  {'-'*4}")
    for r in sorted(results, key=lambda x: -x.score):
        model_label = r.primary_model.split("/")[-1][:23]
        print(
            f"  {r.task_id:<32} {model_label:<24} {r.score:>7.3f}  "
            f"{r.total_reward:>8.2f}  {r.total_completed:>5}  "
            f"{len(r.model_rotations):>4}"
        )
    avg = sum(r.score for r in results) / len(results)
    print(f"  {'-'*32} {'-'*24} {'-'*7}  {'-'*8}  {'-'*5}  {'-'*4}")
    print(f"  {'AVERAGE':<32} {'':<24} {avg:>7.3f}")
    print(f"{'═'*72}\n")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 13 — CLI Entry Point
# ══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Gov Workflow OpenEnv — Multi-Model Rotating LLM Baseline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
10-model pool (April 2026):
  llama-3.3-70b → deepseek-v4-flash → deepseek-r1 → nemotron-3-super →
  qwen3.5-122b → deepseek-v3 → minimax-m2.7 → gemma-4-31b →
  phi-4-mini → llama-3.1-8b

Examples:
  python baseline_openai.py --agent heuristic --verbose
  python baseline_openai.py --agent llm --task district_backlog_easy --verbose
  python baseline_openai.py --agent llm --task all --save-results
  python baseline_openai.py --agent llm --model deepseek-ai/deepseek-v4-flash
  python baseline_openai.py --mode http --url http://localhost:7860 --agent llm
        """,
    )
    p.add_argument("--agent", choices=["llm", "heuristic"], default="heuristic")
    p.add_argument("--task", choices=list_tasks() + ["all"], default="all")
    p.add_argument("--model", default=None)
    p.add_argument("--mode", choices=["direct", "http"], default="direct")
    p.add_argument("--url", default="http://localhost:7860")
    p.add_argument("--max-steps", type=int, default=MAX_LLM_STEPS)
    p.add_argument("--delay", type=float, default=None)
    p.add_argument("--api-key", default=None)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--save-results", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    tasks = list_tasks() if args.task == "all" else [args.task]

    print(f"\n{'═'*65}")
    print("  Gov Workflow OpenEnv — Baseline Runner (April 2026)")
    print(f"  Agent : {args.agent.upper()}")
    if args.agent == "llm":
        pool_disp = " → ".join(m.split("/")[-1][:12] for m in GLOBAL_MODEL_POOL)
        print(f"  Pool  : {pool_disp}")
    print(f"  Mode  : {args.mode}  |  Tasks: {', '.join(tasks)}")
    print(f"{'═'*65}")

    if args.agent == "llm":
        key = args.api_key or os.environ.get("NVIDIA_API_KEY", "")
        if not key:
            print("\n❌  NVIDIA_API_KEY not set.")
            print("    .env file  : NVIDIA_API_KEY=nvapi-xxxx")
            print("    PowerShell : $env:NVIDIA_API_KEY='nvapi-xxxx'")
            print("    Get free key: https://build.nvidia.com/explore/discover\n")
            sys.exit(1)
    else:
        key = None

    results: list[EpisodeResult] = []
    for task_id in tasks:
        result = run_episode(
            task_id=task_id,
            agent_type=args.agent,
            model_override=args.model,
            mode=args.mode,
            server_url=args.url,
            api_key=key,
            verbose=args.verbose,
            max_steps=args.max_steps,
            delay_override=args.delay,
        )
        results.append(result)

    print_leaderboard(results)

    if args.save_results:
        out = save_results(results, Path("results"))
        print(f"  Results saved → {out}\n")


if __name__ == "__main__":
    main()