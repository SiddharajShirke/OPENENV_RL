from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass
from typing import Any, Literal

from openai import OpenAI

from app.baselines import POLICIES, backlog_clearance_policy
from app.env import GovWorkflowEnv
from app.graders import grade_episode
from app.models import ActionModel, ActionType, ObservationModel, PriorityMode, ServiceType

SimulationAgentMode = Literal["baseline_policy", "llm_inference", "trained_rl"]

LEGACY_NVIDIA_MODEL_POOL = [
    "meta/llama-3.3-70b-instruct",
    "qwen/qwen3-next-80b-a3b-instruct",
    "moonshotai/kimi-k2-instruct-0905",
    "meta/llama-3.1-405b-instruct",
    "deepseek-ai/deepseek-v3.2",
    "qwen/qwq-32b",
    "mistralai/mixtral-8x22b-instruct-v0.1",
    "google/gemma-3-27b-it",
    "microsoft/phi-4-mini-instruct",
    "meta/llama-3.1-8b-instruct",
]


@dataclass
class SimulationRun:
    task_id: str
    agent_mode: SimulationAgentMode
    seed: int
    total_reward: float
    score: float
    grader_name: str
    summary: dict[str, Any]
    trace: list[dict[str, Any]]


def _dedupe(values: list[str | None]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value is None:
            continue
        v = value.strip()
        if v and v not in out:
            out.append(v)
    return out


def _env_csv_list(name: str) -> list[str]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def _extract_json_object(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _coerce_action(payload: dict[str, Any] | None) -> ActionModel:
    if not payload:
        return ActionModel(action_type=ActionType.ADVANCE_TIME)
    try:
        return ActionModel(**payload)
    except Exception:
        return ActionModel(action_type=ActionType.ADVANCE_TIME)


def _queue_rows(obs: ObservationModel) -> list[dict[str, Any]]:
    return [
        {
            "service": q.service.value,
            "active_cases": q.active_cases,
            "missing_docs_cases": q.missing_docs_cases,
            "urgent_cases": q.urgent_cases,
            "breached_cases": q.breached_cases,
            "avg_age_days": q.avg_age_days,
        }
        for q in obs.queue_snapshots
    ]


def _alloc_for(obs: ObservationModel, service: ServiceType) -> int:
    raw = obs.officer_pool.allocations.get(service)
    if raw is None:
        raw = obs.officer_pool.allocations.get(service.value, 0)
    return int(raw or 0)


def _top_backlog_service(
    obs: ObservationModel,
    *,
    exclude: ServiceType | None = None,
) -> ServiceType | None:
    ranked = [q for q in obs.queue_snapshots if q.service != exclude]
    if not ranked:
        return None
    ranked.sort(
        key=lambda q: (q.active_cases + (2 * q.breached_cases) + q.urgent_cases, q.avg_age_days),
        reverse=True,
    )
    return ranked[0].service


def _service_with_missing_docs(obs: ObservationModel) -> ServiceType | None:
    candidates = [q for q in obs.queue_snapshots if q.missing_docs_cases > 0]
    if not candidates:
        return None
    candidates.sort(key=lambda q: (q.missing_docs_cases, q.active_cases), reverse=True)
    return candidates[0].service


def _service_with_officers(obs: ObservationModel) -> ServiceType | None:
    services = [q.service for q in obs.queue_snapshots]
    services.sort(key=lambda s: _alloc_for(obs, s), reverse=True)
    for service in services:
        if _alloc_for(obs, service) > 0:
            return service
    return None


def _repair_action_for_observation(
    action: ActionModel,
    obs: ObservationModel,
) -> tuple[ActionModel, str | None]:
    at = action.action_type

    if at == ActionType.ADVANCE_TIME:
        return action, None

    if at == ActionType.SET_PRIORITY_MODE:
        if action.priority_mode is None:
            return (
                ActionModel(action_type=ActionType.SET_PRIORITY_MODE, priority_mode=PriorityMode.BACKLOG_CLEARANCE),
                "missing priority_mode, defaulted to backlog_clearance",
            )
        return action, None

    if at == ActionType.ASSIGN_CAPACITY:
        reserve = int(obs.officer_pool.reserve_officers)
        if reserve <= 0:
            return ActionModel(action_type=ActionType.ADVANCE_TIME), "reserve officers exhausted, switched to advance_time"
        service = action.service or _top_backlog_service(obs)
        if service is None:
            return ActionModel(action_type=ActionType.ADVANCE_TIME), "no service available for assign_capacity"
        delta = int(action.officer_delta) if int(action.officer_delta) > 0 else 1
        delta = min(delta, reserve)
        repaired = ActionModel(
            action_type=ActionType.ASSIGN_CAPACITY,
            service=service,
            officer_delta=delta,
        )
        note = None if repaired.model_dump(exclude_none=True) == action.model_dump(exclude_none=True) else "repaired assign_capacity payload"
        return repaired, note

    if at == ActionType.REQUEST_MISSING_DOCUMENTS:
        service = action.service or _service_with_missing_docs(obs)
        if service is None:
            return ActionModel(action_type=ActionType.ADVANCE_TIME), "no missing-doc queue available, switched to advance_time"
        repaired = ActionModel(
            action_type=ActionType.REQUEST_MISSING_DOCUMENTS,
            service=service,
        )
        note = None if repaired.model_dump(exclude_none=True) == action.model_dump(exclude_none=True) else "repaired request_missing_documents payload"
        return repaired, note

    if at == ActionType.ESCALATE_SERVICE:
        if int(obs.escalation_budget_remaining) <= 0:
            return ActionModel(action_type=ActionType.ADVANCE_TIME), "escalation budget exhausted, switched to advance_time"
        service = action.service or _top_backlog_service(obs)
        if service is None and action.case_id is None:
            return ActionModel(action_type=ActionType.ADVANCE_TIME), "no escalation target available, switched to advance_time"
        repaired = ActionModel(
            action_type=ActionType.ESCALATE_SERVICE,
            service=service,
            case_id=action.case_id,
        )
        note = None if repaired.model_dump(exclude_none=True) == action.model_dump(exclude_none=True) else "repaired escalate_service payload"
        return repaired, note

    if at == ActionType.REALLOCATE_OFFICERS:
        source = action.service or _service_with_officers(obs)
        if source is None:
            return ActionModel(action_type=ActionType.ADVANCE_TIME), "no staffed source service, switched to advance_time"
        source_alloc = _alloc_for(obs, source)
        if source_alloc <= 0:
            source = _service_with_officers(obs)
            source_alloc = _alloc_for(obs, source) if source is not None else 0
        if source is None or source_alloc <= 0:
            return ActionModel(action_type=ActionType.ADVANCE_TIME), "insufficient source officers, switched to advance_time"

        target = action.target_service
        if target is None or target == source:
            target = _top_backlog_service(obs, exclude=source)
        if target is None or target == source:
            return ActionModel(action_type=ActionType.ADVANCE_TIME), "missing distinct target_service, switched to advance_time"

        delta = int(action.officer_delta) if int(action.officer_delta) > 0 else 1
        delta = max(1, min(delta, source_alloc))
        repaired = ActionModel(
            action_type=ActionType.REALLOCATE_OFFICERS,
            service=source,
            target_service=target,
            officer_delta=delta,
        )
        note = None if repaired.model_dump(exclude_none=True) == action.model_dump(exclude_none=True) else "repaired reallocate_officers payload"
        return repaired, note

    return action, None


def _model_label_for_mode(agent_mode: SimulationAgentMode) -> str:
    if agent_mode == "baseline_policy":
        return "baseline_policy"
    if agent_mode == "trained_rl":
        return "trained_rl"
    return os.getenv("MODEL_NAME", "llm_inference")


def _log_step_line(step_row: dict[str, Any]) -> str:
    done = "true" if bool(step_row.get("done")) else "false"
    error = step_row.get("last_action_error") or "null"
    action = json.dumps(step_row.get("action_payload", {}), separators=(",", ":"))
    source = step_row.get("decision_source") or "unknown"
    model = step_row.get("model_used") or "null"
    repair = step_row.get("repair_note") or "null"
    return (
        f"[STEP] step={step_row.get('step', 0)} action={action} "
        f"reward={float(step_row.get('reward', 0.0)):.2f} done={done} "
        f"error={error} source={source} model={model} repair={repair}"
    )


class LiveSimulationSession:
    def __init__(
        self,
        *,
        task_id: str,
        agent_mode: SimulationAgentMode,
        max_steps: int,
        seed: int | None,
        policy_name: str | None = None,
        model_path: str | None = None,
        model_type: Literal["maskable", "recurrent"] = "maskable",
    ) -> None:
        self.task_id = task_id
        self.agent_mode = agent_mode
        self.max_steps = max_steps
        self.seed = int(seed if seed is not None else random.randint(1, 999999))
        self.policy_name = policy_name or "backlog_clearance"
        self.model_path = model_path
        self.model_type = model_type
        self.trace: list[dict[str, Any]] = []
        self.total_reward = 0.0
        self.step_idx = 0
        self.done = False
        self.summary: dict[str, Any] | None = None
        self.score: float | None = None
        self.grader_name: str | None = None

        self.env: GovWorkflowEnv | None = None
        self.obs: ObservationModel | Any = None
        self.policy = None

        self.rl_env: Any = None
        self.rl_model: Any = None
        self.rl_lstm_state: Any = None
        self.rl_episode_start: Any = None

        self.llm_runtimes: list[dict[str, Any]] = []
        self.llm_route: list[str] = []

        if self.agent_mode == "trained_rl":
            self._init_trained()
        else:
            self._init_core()

    def start_line(self) -> str:
        return (
            f"[START] task={self.task_id} env=gov-workflow-openenv "
            f"model={_model_label_for_mode(self.agent_mode)}"
        )

    def _init_core(self) -> None:
        self.env = GovWorkflowEnv(task_id=self.task_id)
        self.obs, _ = self.env.reset(seed=self.seed)
        if self.agent_mode == "baseline_policy":
            self.policy = POLICIES.get(self.policy_name, backlog_clearance_policy)
        else:
            self.policy = self._llm_action_with_meta
            self._init_llm_runtimes()

    def _init_llm_runtimes(self) -> None:
        openai_base = os.getenv("API_BASE_URL") or os.getenv("OPENAI_API_BASE_URL") or "https://api.openai.com/v1"
        nvidia_base = os.getenv("NVIDIA_API_BASE_URL", "https://integrate.api.nvidia.com/v1")

        openai_keys = _dedupe(
            [
                os.getenv("HF_TOKEN"),
                os.getenv("OPENAI_API_KEY"),
                os.getenv("API_KEY"),
            ]
        )
        nvidia_keys = _dedupe(
            [
                os.getenv("NVIDIA_API_KEY"),
                os.getenv("NVIDIA_API_KEY_2"),
            ]
        )

        openai_models = _dedupe(
            [
                os.getenv("MODEL_NAME", "meta/llama-3.3-70b-instruct"),
                *_env_csv_list("MODEL_FALLBACKS"),
            ]
        )
        nvidia_models = _dedupe(
            [
                os.getenv("NVIDIA_MODEL"),
                *_env_csv_list("NVIDIA_MODEL_FALLBACKS"),
                *LEGACY_NVIDIA_MODEL_POOL,
            ]
        )

        runtimes: list[dict[str, Any]] = []

        if openai_keys and openai_models:
            clients: list[tuple[OpenAI, str]] = []
            for idx, key in enumerate(openai_keys, start=1):
                try:
                    clients.append((OpenAI(base_url=openai_base, api_key=key, timeout=8.0, max_retries=0), f"openai_key_{idx}"))
                except Exception:
                    continue
            if clients:
                runtimes.append(
                    {
                        "provider": "openai-compatible",
                        "base_url": openai_base,
                        "clients": clients,
                        "models": openai_models,
                    }
                )

        if nvidia_keys and nvidia_models:
            clients = []
            for idx, key in enumerate(nvidia_keys, start=1):
                try:
                    clients.append((OpenAI(base_url=nvidia_base, api_key=key, timeout=8.0, max_retries=0), f"nvidia_key_{idx}"))
                except Exception:
                    continue
            if clients:
                runtimes.append(
                    {
                        "provider": "nvidia",
                        "base_url": nvidia_base,
                        "clients": clients,
                        "models": nvidia_models,
                    }
                )

        self.llm_runtimes = runtimes

        openai_runtime = next((rt for rt in runtimes if rt.get("provider") == "openai-compatible"), None)
        nvidia_runtime = next((rt for rt in runtimes if rt.get("provider") == "nvidia"), None)

        if openai_runtime is not None:
            openai_route = (
                f"openai-compatible ({len(openai_runtime['clients'])} keys, "
                f"{len(openai_runtime['models'])} models)"
            )
        else:
            openai_route = "openai-compatible (unavailable: missing API key/model)"

        if nvidia_runtime is not None:
            nvidia_route = (
                f"nvidia ({len(nvidia_runtime['clients'])} keys, "
                f"{len(nvidia_runtime['models'])} models)"
            )
        else:
            nvidia_route = "nvidia (unavailable: missing API key/model)"

        self.llm_route = [
            openai_route,
            nvidia_route,
            "heuristic fallback (backlog_clearance_policy)",
        ]

    def _llm_action_with_meta(self, obs: ObservationModel) -> tuple[ActionModel, dict[str, Any]]:
        attempts = 0
        last_error = ""
        system_prompt = (
            "You are controlling a government workflow simulator. "
            "Return exactly one JSON object only. No markdown. No explanation. "
            "Allowed action_type: set_priority_mode, assign_capacity, request_missing_documents, "
            "escalate_service, advance_time, reallocate_officers. "
            "Rules: "
            "1) reallocate_officers requires service + target_service + officer_delta>0 and source!=target. "
            "2) assign_capacity requires service + officer_delta>0. "
            "3) request_missing_documents requires service with missing_docs_cases>0. "
            "4) set_priority_mode requires priority_mode in [urgent_first, oldest_first, balanced, backlog_clearance]. "
            "Use lowercase enum values."
        )
        user_prompt = (
            "Observation:\n"
            f"{obs.model_dump_json()}\n"
            f"Last action validity: {obs.last_action_valid}\n"
            f"Last action message: {obs.last_action_message}\n"
            "Return action JSON."
        )

        for runtime in self.llm_runtimes:
            for client, key_label in runtime["clients"]:
                for model in runtime["models"]:
                    attempts += 1
                    try:
                        out = client.chat.completions.create(
                            model=model,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt},
                            ],
                            temperature=0.0,
                            max_tokens=200,
                            stream=False,
                        )
                        content = (out.choices[0].message.content or "").strip()
                        action = _coerce_action(_extract_json_object(content))
                        return action, {
                            "decision_source": "llm",
                            "provider": runtime["provider"],
                            "model_used": model,
                            "llm_attempts": attempts,
                            "llm_error": None,
                            "llm_key_label": key_label,
                        }
                    except Exception as exc:
                        last_error = str(exc)
                        continue

        action = backlog_clearance_policy(obs)
        if not self.llm_runtimes:
            last_error = "No LLM credentials configured."
        return action, {
            "decision_source": "heuristic_fallback",
            "provider": "heuristic",
            "model_used": "backlog_clearance_policy",
            "llm_attempts": attempts,
            "llm_error": last_error or None,
            "llm_key_label": None,
        }

    def _init_trained(self) -> None:
        import numpy as np
        from app.main import _load_model_cached_or_503, _resolve_model_path_or_422
        from rl.gym_wrapper import GovWorkflowGymEnv

        if not self.model_path:
            raise ValueError("model_path is required for trained_rl simulation.")
        model_abs = _resolve_model_path_or_422(self.model_path)
        self.rl_model = _load_model_cached_or_503(model_abs, self.model_type)
        self.rl_env = GovWorkflowGymEnv(task_id=self.task_id, seed=self.seed, hard_action_mask=True)
        self.obs, _ = self.rl_env.reset(seed=self.seed)
        self.rl_lstm_state = None
        self.rl_episode_start = np.array([True], dtype=bool)

    def step_once(self) -> tuple[dict[str, Any], str, bool]:
        if self.done:
            raise RuntimeError("Simulation already finished.")

        self.step_idx += 1
        if self.agent_mode == "trained_rl":
            row = self._step_trained()
        else:
            row = self._step_core()
        self.trace.append(row)
        self.total_reward += float(row["reward"])
        step_log = _log_step_line(row)

        if row["done"] or self.step_idx >= self.max_steps:
            self._finalize()
            row["done"] = True
            return row, step_log, True
        return row, step_log, False

    def end_line(self) -> str:
        if self.score is None:
            return "[END] success=false steps=0 score=0.00 rewards="
        rewards = ",".join(f"{float(x.get('reward', 0.0)):.2f}" for x in self.trace)
        success = "true" if self.score >= 0.5 else "false"
        return (
            f"[END] success={success} steps={len(self.trace)} "
            f"score={self.score:.2f} rewards={rewards}"
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "agent_mode": self.agent_mode,
            "seed": self.seed,
            "max_steps": self.max_steps,
            "step_idx": self.step_idx,
            "done": self.done,
            "total_reward": float(self.total_reward),
            "score": self.score,
            "grader_name": self.grader_name,
            "summary": self.summary,
            "trace_len": len(self.trace),
            "llm_route": list(self.llm_route),
        }

    def close(self) -> None:
        try:
            if self.env is not None and hasattr(self.env, "close"):
                self.env.close()
        except Exception:
            pass
        try:
            if self.rl_env is not None and hasattr(self.rl_env, "close"):
                self.rl_env.close()
        except Exception:
            pass

    def _step_core(self) -> dict[str, Any]:
        if self.env is None:
            raise RuntimeError("Core simulation env not initialized.")
        if self.agent_mode == "baseline_policy":
            action = self.policy(self.obs)
            meta = {
                "decision_source": "baseline_policy",
                "provider": "local_policy",
                "model_used": self.policy_name,
                "llm_attempts": 0,
                "llm_error": None,
                "llm_key_label": None,
            }
        else:
            action, meta = self.policy(self.obs)
            repaired_action, repair_note = _repair_action_for_observation(action, self.obs)
            if repair_note:
                action = repaired_action
                if meta.get("decision_source") == "llm":
                    meta["decision_source"] = "llm_repaired"
                meta["repair_note"] = repair_note

        self.obs, reward, terminated, truncated, info = self.env.step(action)
        done = bool(terminated or truncated)
        row = {
            "step": self.step_idx,
            "day": self.obs.day,
            "action_type": action.action_type.value,
            "action_payload": action.model_dump(exclude_none=True, mode="json"),
            "reward": float(reward),
            "done": done,
            "backlog": self.obs.total_backlog,
            "completed": self.obs.total_completed,
            "sla_breaches": self.obs.total_sla_breaches,
            "fairness_gap": float(self.obs.fairness_gap),
            "escalation_budget_remaining": self.obs.escalation_budget_remaining,
            "invalid_action": bool(info.invalid_action),
            "last_action_error": info.last_action_error,
            "queue_rows": _queue_rows(self.obs),
        }
        row.update(meta)
        return row

    def _step_trained(self) -> dict[str, Any]:
        import numpy as np

        masks = self.rl_env.action_masks()
        if self.model_type == "recurrent":
            action, self.rl_lstm_state = self.rl_model.predict(
                self.obs,
                state=self.rl_lstm_state,
                episode_start=self.rl_episode_start,
                deterministic=True,
            )
            action_idx = int(action.item() if hasattr(action, "item") else action)
            if not (0 <= action_idx < masks.shape[0] and bool(masks[action_idx])):
                valid = np.flatnonzero(masks)
                action_idx = int(valid[0]) if valid.size > 0 else 18
        else:
            from sb3_contrib.common.maskable.utils import get_action_masks

            action, _ = self.rl_model.predict(
                self.obs,
                action_masks=get_action_masks(self.rl_env),
                deterministic=True,
            )
            action_idx = int(action.item() if hasattr(action, "item") else action)

        self.obs, reward, terminated, truncated, info = self.rl_env.step(action_idx)
        done = bool(terminated or truncated)
        if self.model_type == "recurrent":
            self.rl_episode_start = np.array([done], dtype=bool)
        core_obs = self.rl_env._core_env._build_observation()
        action_model, action_label = _decode_action_idx(action_idx)
        return {
            "step": self.step_idx,
            "day": core_obs.day,
            "action_type": action_label,
            "action_payload": action_model.model_dump(exclude_none=True, mode="json"),
            "action_index": action_idx,
            "reward": float(reward),
            "done": done,
            "backlog": core_obs.total_backlog,
            "completed": core_obs.total_completed,
            "sla_breaches": core_obs.total_sla_breaches,
            "fairness_gap": float(core_obs.fairness_gap),
            "escalation_budget_remaining": core_obs.escalation_budget_remaining,
            "invalid_action": bool(info.get("invalid_action", False)),
            "last_action_error": info.get("last_action_error"),
            "queue_rows": _queue_rows(core_obs),
            "decision_source": "trained_rl",
            "provider": "rl",
            "model_used": self.model_path or "trained_rl",
            "llm_attempts": 0,
            "llm_error": None,
            "llm_key_label": None,
        }

    def _finalize(self) -> None:
        if self.done:
            return
        self.done = True
        if self.agent_mode == "trained_rl":
            final_state = self.rl_env._core_env.state()
        else:
            final_state = self.env.state()
        gr = grade_episode(final_state)
        self.score = float(gr.score)
        self.grader_name = gr.grader_name

        llm_steps = sum(
            1 for row in self.trace if row.get("decision_source") in {"llm", "llm_repaired"}
        )
        fallback_steps = sum(1 for row in self.trace if row.get("decision_source") == "heuristic_fallback")
        repaired_steps = sum(1 for row in self.trace if row.get("decision_source") == "llm_repaired")

        self.summary = {
            "total_steps": final_state.total_steps,
            "total_completed": final_state.total_completed,
            "total_backlog": final_state.total_backlog,
            "total_sla_breaches": final_state.total_sla_breaches,
            "fairness_gap": float(final_state.fairness_gap),
            "total_invalid_actions": final_state.metrics.total_invalid_actions,
            "llm_steps": llm_steps,
            "heuristic_fallback_steps": fallback_steps,
            "llm_repaired_steps": repaired_steps,
        }
        if self.agent_mode == "llm_inference":
            self.summary["llm_route"] = list(self.llm_route)
        if self.agent_mode == "trained_rl":
            self.summary["model_path"] = self.model_path
            self.summary["model_type"] = self.model_type


def run_simulation(
    *,
    task_id: str,
    agent_mode: SimulationAgentMode,
    max_steps: int,
    seed: int | None,
    policy_name: str | None = None,
    model_path: str | None = None,
    model_type: Literal["maskable", "recurrent"] = "maskable",
) -> SimulationRun:
    session = LiveSimulationSession(
        task_id=task_id,
        agent_mode=agent_mode,
        max_steps=max_steps,
        seed=seed,
        policy_name=policy_name,
        model_path=model_path,
        model_type=model_type,
    )
    try:
        while not session.done:
            session.step_once()
        return SimulationRun(
            task_id=session.task_id,
            agent_mode=session.agent_mode,
            seed=session.seed,
            total_reward=float(session.total_reward),
            score=float(session.score or 0.0),
            grader_name=str(session.grader_name or "unknown"),
            summary=dict(session.summary or {}),
            trace=list(session.trace),
        )
    finally:
        session.close()


def _decode_action_idx(action_idx: int) -> tuple[ActionModel, str]:
    try:
        from rl.feature_builder import ACTION_DECODE_TABLE
        from app.models import PriorityMode, ServiceType
    except Exception:
        return ActionModel(action_type=ActionType.ADVANCE_TIME), f"action_{action_idx}"

    row = ACTION_DECODE_TABLE.get(int(action_idx))
    if row is None:
        return ActionModel(action_type=ActionType.ADVANCE_TIME), f"action_{action_idx}"

    action_type, service, priority_mode, delta = row
    kwargs: dict[str, Any] = {"action_type": action_type}
    if service is not None:
        kwargs["service"] = service
    if priority_mode is not None:
        kwargs["priority_mode"] = priority_mode
    if delta is not None:
        kwargs["officer_delta"] = int(delta)
    try:
        if isinstance(kwargs.get("service"), str):
            kwargs["service"] = ServiceType(kwargs["service"])
        if isinstance(kwargs.get("priority_mode"), str):
            kwargs["priority_mode"] = PriorityMode(kwargs["priority_mode"])
        action = ActionModel(**kwargs)
    except Exception:
        action = ActionModel(action_type=ActionType.ADVANCE_TIME)
    return action, action_type
