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

# ── Confirmed model IDs on NVIDIA Build API (April 2026) ─────────────────────

# ── Global 10-Model Sequential Pool ───────────────────────────────────────────
# Consistent order for all tasks (easy, medium, hard).
# Fallback rotates progressively from primary → highest cap → fastest.
GLOBAL_MODEL_POOL: list[str] = [
    "meta/llama-3.3-70b-instruct",         # 1. Primary
    "qwen/qwen3-next-80b-a3b-instruct",    # 2. Reasoning
    "moonshotai/kimi-k2-instruct-0905",    # 3. Planning
    "meta/llama-3.1-405b-instruct",        # 4. Max Capacity
    "deepseek-ai/deepseek-v3.2",           # 5. High Perf
    "qwen/qwq-32b",                        # 6. Thinking Fallback
    "mistralai/mixtral-8x22b-instruct-v0.1",# 7. Fast MoE
    "google/gemma-3-27b-it",               # 8. Lightweight
    "microsoft/phi-4-mini-instruct",       # 9. Reliable Last Resort
    "meta/llama-3.1-8b-instruct"           # 10. Fast Safety Fallback
]

# ── Free endpoint pool (KEY 2 fallback) ───────────────────────────────────────
# Separated from primary bucket. These are "FREE" tagged models on NVIDIA Build.
FREE_POOL: list[str] = [
    "microsoft/phi-4-mini-instruct",
    "meta/llama-3.1-8b-instruct"
]

# ── Fixed seeds (one per task for reproducibility) ────────────────────────────
TASK_SEEDS: dict[str, int] = {
    "district_backlog_easy": 11,
    "mixed_urgency_medium":  22,
    "cross_department_hard": 33,
}

# ── LLM generation settings ───────────────────────────────────────────────────
LLM_TEMPERATURE = 0.2    # low = deterministic, structured JSON output
LLM_TOP_P       = 0.7
LLM_MAX_TOKENS  = 512
MAX_LLM_STEPS   = 80

# ── Rate-limit & retry settings ───────────────────────────────────────────────
# NVIDIA free tier enforces ~5 RPM = 1 call every 12 seconds.
LLM_CALL_DELAY  = float(os.environ.get("LLM_CALL_DELAY", "12.0"))
LLM_CALL_JITTER = 1.0    # ± random jitter added to avoid thundering-herd

# ── Enum fields that must be lowercase for Pydantic ───────────────────────────
_ENUM_FIELDS = {"action_type", "priority_mode", "service", "target_service"}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Model Rotator
# ══════════════════════════════════════════════════════════════════════════════

class ModelRotator:
    """
    Manages automatic model rotation through the global fallback sequence.

    Lifecycle per episode:
      1. Starts at the 1st model in the 10-model sequence.
      2. If an error occurs, the process instantly ends for that model, and it 
         rotates to the next model cyclically.
    """

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
        """Returns 2 if it's a free model, 1 otherwise."""
        if self.current in FREE_POOL:
            return 2
        return 1

    @property
    def pool_exhausted(self) -> bool:
        # We rotate cyclically, so the pool is never permanently exhausted.
        # But we track if we did a full loop without any progress.
        return len(self._rotation_log) >= 50

    def rotate(self, reason: str = "error") -> str | None:
        """
        Advance to the next model in the list in an endless loop.
        """
        old = self.current
        self._rotation_log.append({
            "from": old,
            "reason": reason,
        })
        
        self._index = (self._index + 1) % len(self._sequence)
        new = self._sequence[self._index]
        
        print(
            f"\n  🔄 Model rotated [Fallback]: "
            f"{old}  →  {new}"
            f"  (reason: {reason})"
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
        # Compute specific model performance/usage counts for final report
        usage = {}
        for r in self.step_log:
            usage[r.model_used] = usage.get(r.model_used, 0) + 1
        usage_str = ", ".join(f"{m} ({c} steps)" for m, c in usage.items())
        
        return (
            f"[{self.task_id}] agent={self.agent} "
            f"score={self.score:.3f} reward={self.total_reward:.2f} "
            f"completed={self.total_completed} breaches={self.total_sla_breaches} "
            f"invalid={self.total_invalid_actions} "
            f"rotations={len(self.model_rotations)} "
            f"day={self.final_day} steps={self.total_steps} "
            f"time={self.elapsed_seconds:.1f}s\n"
            f"Model Performance Use: {usage_str}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Direct Environment Wrapper
# ══════════════════════════════════════════════════════════════════════════════

class DirectEnvClient:
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
        result = grade_episode(self.env.state())
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
    Useful as score floor and for offline/CI testing.

    Strategy (priority order every step):
      1. Day 0: set priority mode → sla_aware
      2. Day 0: deploy reserve officers to most backlogged service
      3. Any day: resolve missing documents (highest count first)
      4. Final 5 days + urgent backlog: escalate one case
      5. Load imbalance > 3x: reallocate one officer
      6. Default: advance_time
    """

    def __init__(self) -> None:
        self._priority_set = False
        self._reserve_deployed = False

    def reset(self) -> None:
        self._priority_set = False
        self._reserve_deployed = False

    # HeuristicAgent has no current_model — expose for run_episode compat
    current_model = "heuristic"

    def rotation_summary(self) -> list[dict]:
        return []

    def update_reward(self, _: float) -> None:
        pass

    def act(self, obs: ObservationModel) -> ActionModel:
        if not self._priority_set:
            self._priority_set = True
            return ActionModel(
                action_type=ActionType.SET_PRIORITY_MODE,
                priority_mode=PriorityMode.URGENT_FIRST,
            )

        if not self._reserve_deployed and obs.officer_pool.reserve_officers > 0:
            self._reserve_deployed = True
            most_loaded = max(obs.queue_snapshots, key=lambda s: s.active_cases)
            return ActionModel(
                action_type=ActionType.ASSIGN_CAPACITY,
                service=most_loaded.service,
                officer_delta=1,
            )

        for snap in sorted(obs.queue_snapshots, key=lambda s: -s.missing_docs_cases):
            if snap.missing_docs_cases > 0:
                return ActionModel(
                    action_type=ActionType.REQUEST_MISSING_DOCUMENTS,
                    service=snap.service,
                )

        if obs.escalation_budget_remaining > 0:
            urgent_snaps = [s for s in obs.queue_snapshots if s.urgent_cases > 0]
            if urgent_snaps and (obs.max_days - obs.day) <= 5:
                target = max(urgent_snaps, key=lambda s: s.urgent_cases)
                return ActionModel(
                    action_type=ActionType.ESCALATE_SERVICE,
                    service=target.service,
                )

        if obs.officer_pool.allocations:
            case_counts = {s.service: s.active_cases for s in obs.queue_snapshots}
            for src_svc, src_off in obs.officer_pool.allocations.items():
                if src_off < 2:
                    continue
                src_load = case_counts.get(src_svc, 0) / max(src_off, 1)
                for tgt_svc, tgt_off in obs.officer_pool.allocations.items():
                    if tgt_svc == src_svc:
                        continue
                    tgt_load = case_counts.get(tgt_svc, 0) / max(tgt_off, 1)
                    if tgt_load > src_load * 3:
                        return ActionModel(
                            action_type=ActionType.REALLOCATE_OFFICERS,
                            service=src_svc,
                            target_service=tgt_svc,
                            officer_delta=1,
                        )

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
  - advance_time is the ONLY action that earns progress reward — call it every day
  - Do NOT chain more than 2 admin actions before calling advance_time
  - Do NOT escalate before (max_days - 5) unless case already breached SLA
  - Do NOT reallocate if source service has fewer than 2 officers

OPTIMAL STRATEGY:
  Day 0:    set_priority_mode → assign_capacity (if reserves > 0) → advance_time
  Every day: request_missing_documents (ONE service only) → advance_time
  Final 5:  escalate_service (urgent/breached only) → advance_time
  MAXIMISE advance_time calls — that is where all completions happen.

RESPONSE FORMAT — return ONLY a raw JSON object, nothing else:
  CORRECT:   {"action_type": "advance_time"}
  CORRECT:   {"action_type": "request_missing_documents", "service": "driving_license"}
  WRONG:     ```json\\n{"action_type": "ADVANCE_TIME"}```
"""


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — JSON Extraction with Lowercase Normaliser  [FIX 1]
# ══════════════════════════════════════════════════════════════════════════════

def _extract_json_action(raw: str) -> dict[str, Any]:
    """
    Extracts JSON from LLM response and normalises all enum fields to lowercase.

    FIX 1: LLMs (especially instruction-tuned models) return DRIVING_LICENSE,
    SET_PRIORITY_MODE, ADVANCE_TIME etc. in UPPERCASE.
    Pydantic StrEnum expects lowercase snake_case values.
    Normalise silently here so ActionModel(**parsed) never fails on case alone.
    """
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
        print(
            f"  ⚠ Could not parse LLM JSON, falling back to advance_time. "
            f"Raw snippet: {raw[:120]}"
        )
        return {"action_type": "advance_time"}

    # Normalise enum fields: DRIVING_LICENSE → driving_license
    for enum_field in _ENUM_FIELDS:
        if enum_field in parsed and isinstance(parsed[enum_field], str):
            parsed[enum_field] = parsed[enum_field].lower()

    return parsed


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — LLM Agent with Model Rotation  [FIX 2 + FIX 3]
# ══════════════════════════════════════════════════════════════════════════════

def _build_user_message(
    obs: ObservationModel, step_num: int, cumulative_reward: float
) -> str:
    queue_lines = []
    for snap in obs.queue_snapshots:
        officers = obs.officer_pool.allocations.get(snap.service, 0)
        queue_lines.append(
            f"  {snap.service}: backlog={snap.active_cases} "
            f"officers={officers} missing_docs={snap.missing_docs_cases} "
            f"urgent={snap.urgent_cases} breached={snap.breached_cases} "
            f"avg_age={snap.avg_age_days:.1f}d"
        )
    return (
        f"STEP {step_num} | Day {obs.day}/{obs.max_days} "
        f"| Days remaining: {obs.max_days - obs.day}\n"
        f"Cumulative reward: {cumulative_reward:.2f}\n"
        f"Priority mode: {obs.priority_mode}\n"
        f"Reserve officers available: {obs.officer_pool.reserve_officers}\n"
        f"Escalation budget remaining: {obs.escalation_budget_remaining}\n"
        f"Total backlog: {obs.total_backlog} | Completed: {obs.total_completed} "
        f"| SLA breaches: {obs.total_sla_breaches}\n"
        f"Fairness gap: {obs.fairness_gap:.3f}\n\n"
        f"QUEUE STATUS:\n" + "\n".join(queue_lines) + "\n\n"
        f"Return a single JSON action object. All values lowercase."
    )


class LLMAgent:
    """
    LLM-powered agent with automatic model rotation on rate-limit errors.

    FIX 2 — Retry + rotate:
      On 429 RateLimitError: wait LLM_RETRY_BASE * attempt seconds.
      After LLM_MAX_RETRIES failures on the same model → rotate via ModelRotator.
      Primary pool is tried first; backup pool activates when primary is exhausted.
      If entire pool exhausted → returns ActionType.ADVANCE_TIME safely.

    FIX 3 — Inter-call throttle:
      12s delay (+/- 1s jitter) between every LLM API call to stay under 5 RPM.
      Applied in run_episode() after every step, not inside act() itself.
    """

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
                "  .env file    : NVIDIA_API_KEY=nvapi-xxxxxxxxxxxx\n"
                "  Get free key : https://build.nvidia.com/explore/discover"
            )

        self._api_key = resolved_key
        self._task_id = task_id
        self._rotator = ModelRotator(task_id)

        # Manual model override (CLI --model flag) goes to front of sequence
        if model_override:
            seq = [model_override] + [
                m for m in self._rotator._sequence if m != model_override
            ]
            self._rotator._sequence = seq

        # We maintain two client instances for rotation
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
        # Entire pool exhausted — safe no-op fallback
        if self._rotator.pool_exhausted:
            return ActionModel(action_type=ActionType.ADVANCE_TIME)

        user_message = _build_user_message(obs, step_num, self._cumulative_reward)
        self._history.append({"role": "user", "content": user_message})

        # Rolling window — keep last 10 exchanges (20 messages)
        if len(self._history) > 20:
            self._history = self._history[-20:]

        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self._history

        raw_reply = ""

        # Loop until a model responds successfully. Replaces old retry/backoff logic.
        while True:
            try:
                # Select correct client based on model pool
                active_client = self._client
                if self._rotator.current_key_id == 2 and self._client_2:
                    active_client = self._client_2

                response = active_client.chat.completions.create(
                    model=self._rotator.current,
                    messages=messages,
                    temperature=LLM_TEMPERATURE,
                    top_p=LLM_TOP_P,
                    max_tokens=LLM_MAX_TOKENS,
                    timeout=30,       # hard 30s socket timeout per call
                )
                raw_reply = response.choices[0].message.content or ""
                break               # success — exit loop

            except KeyboardInterrupt:
                raise               # propagate Ctrl+C immediately

            except Exception as exc:
                err_name = type(exc).__name__
                err_msg  = str(exc)[:120]
                
                print(
                    f"  ⚠ {err_name} on {self._rotator.current}: {err_msg}"
                )
                
                # End process, rotate immediately, repeat same step with new model
                self._rotator.rotate(reason=err_name)
                
                # Sleep briefly to not spam the endpoint infinitely while rotating
                time.sleep(1.0)

        self._history.append({"role": "assistant", "content": raw_reply})

        # FIX 1: normalise UPPERCASE enum values before Pydantic validation
        action_dict = _extract_json_action(raw_reply)

        try:
            return ActionModel(**action_dict)
        except Exception as exc:
            print(f"  ⚠ ActionModel parse failed ({exc}), using advance_time")
            return ActionModel(action_type=ActionType.ADVANCE_TIME)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — Episode Runner
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
    """
    Run a single episode with the specified agent and mode.
    """
    seed = TASK_SEEDS[task_id]
    
    # ── Resolve delay ────────────────────────────────────────────────────────
    # CLI override > Env Var > Global Constant
    delay = delay_override if delay_override is not None else LLM_CALL_DELAY

    # Build environment client
    if mode == "http":
        client: DirectEnvClient | HttpEnvClient = HttpEnvClient(
            task_id, seed, base_url=server_url
        )
    else:
        client = DirectEnvClient(task_id, seed)

    # Build agent
    if agent_type == "llm":
        agent: HeuristicAgent | LLMAgent = LLMAgent(
            task_id=task_id,
            model_override=model_override,
            api_key=api_key,
        )
        primary_label = agent.current_model
        pool = GLOBAL_MODEL_POOL.copy()
    else:
        agent = HeuristicAgent()
        primary_label = "heuristic"
        pool = []

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
        key1_status = "loaded" if os.environ.get("NVIDIA_API_KEY", "") else "MISSING"
        key2_status = "loaded" if os.environ.get("NVIDIA_API_KEY_2", "") else "not set (skipping free pool)"
        print(f"  🔑 KEY 1: {key1_status}  |  ✅ KEY 2: {key2_status}")
        
        print(f"  Pool  : {GLOBAL_MODEL_POOL}")
        if os.environ.get("NVIDIA_API_KEY_2", ""):
            print(f"  Free  : {FREE_POOL}")
    print(f"  Agent : {agent_type}  |  Mode: {mode}  |  Seed: {seed}")
    print(f"  Max steps: {max_steps}  |  Delay: {delay}s")
    print(f"{'═'*65}")

    while not (client.terminated or client.truncated) and step_num < max_steps:
        step_num += 1
        current_model = agent.current_model

        if agent_type == "llm":
            action = agent.act(obs, step_num)           # type: ignore[arg-type]
        else:
            action = agent.act(obs)                     # type: ignore[arg-type]

        obs, reward, terminated, truncated, info = client.step(action)
        agent.update_reward(reward)

        total_reward += reward
        if info.invalid_action:
            total_invalid += 1

        record = StepRecord(
            step=step_num,
            day=obs.day,
            action_type=action.action_type.value,
            reward=round(reward, 4),
            invalid=info.invalid_action,
            total_backlog=obs.total_backlog,
            total_completed=obs.total_completed,
            model_used=current_model,
            notes=info.notes,
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
            if info.notes:
                print(f"         notes: {info.notes}")

        # FIX 3: throttle LLM calls — NVIDIA free tier ≈ 5 RPM (1 call / 12s)
        # Jitter avoids synchronized burst if multiple episodes run concurrently.
        if agent_type == "llm":
            actual_delay = delay + _random.uniform(-LLM_CALL_JITTER, LLM_CALL_JITTER)
            if not verbose:
                print(f"  Executing step {step_num}/{max_steps}... [Sleeping {actual_delay:.1f}s to respect rate limits]", end="\r", flush=True)
            time.sleep(max(1.0, actual_delay))
            if not verbose:
                 print(" " * 80, end="\r", flush=True) # clear the line


    # Grade the episode
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
        print(f"    {metric:<32} {value:.3f}  {bar}")
    if rotations:
        print(f"  Model rotation log:")
        for r in rotations:
            print(
                f"    {r['from']:<40} → rotated ({r['reason']})"
            )
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
# SECTION 11 — Reporter
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
# SECTION 12 — CLI Entry Point
# ══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Gov Workflow OpenEnv — Multi-Model Rotating LLM Baseline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Model sequence (automatic rotation on API failure):
  meta/llama-3.3-70b-instruct → qwen/qwen3-next-80b-a3b-instruct → ... (10 models)

Examples:
  python baseline_openai.py --agent heuristic --verbose
  python baseline_openai.py --agent llm --task district_backlog_easy --verbose
  python baseline_openai.py --agent llm --task all --save-results
  python baseline_openai.py --agent llm --model mistralai/mistral-large-2-instruct
  python baseline_openai.py --mode http --url http://localhost:7860 --agent llm
        """,
    )
    p.add_argument(
        "--agent", choices=["llm", "heuristic"], default="heuristic",
        help="'heuristic' needs no API key. (default: heuristic)",
    )
    p.add_argument(
        "--task", choices=list_tasks() + ["all"], default="all",
        help="Which task to run. 'all' runs all three. (default: all)",
    )
    p.add_argument(
        "--model", default=None,
        help=(
            "Override the primary model for all tasks. "
            "If omitted, each task uses its own pool."
        ),
    )
    p.add_argument("--mode", choices=["direct", "http"], default="direct")
    p.add_argument("--url", default="http://localhost:7860")
    p.add_argument("--max-steps", type=int, default=MAX_LLM_STEPS)
    p.add_argument("--delay", type=float, default=None, help="Override LLM_CALL_DELAY (sec)")
    p.add_argument("--api-key", default=None)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--save-results", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    tasks = list_tasks() if args.task == "all" else [args.task]

    print(f"\n{'═'*65}")
    print("  Gov Workflow OpenEnv — Baseline Runner")
    print(f"  Agent : {args.agent.upper()}")
    if args.agent == "llm":
        if args.model:
            print(f"  Model : {args.model}  (manual override)")
        else:
            print("  Model : Global 10-model fallback sequence")
            print(f"  pool  → {' → '.join(m.split('/')[-1][:15] for m in GLOBAL_MODEL_POOL)}")
    print(f"  Mode  : {args.mode}  |  Tasks: {', '.join(tasks)}")
    print(f"{'═'*65}")

    if args.agent == "llm":
        key = args.api_key or os.environ.get("NVIDIA_API_KEY", "")
        if not key:
            print("\n❌  NVIDIA_API_KEY not set.")
            print("    .env file  : NVIDIA_API_KEY=nvapi-xxxx")
            print("    PowerShell : $env:NVIDIA_API_KEY='nvapi-xxxx'")
            print("    CMD        : set NVIDIA_API_KEY=nvapi-xxxx")
            print("    Free key   : https://build.nvidia.com/explore/discover\n")
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