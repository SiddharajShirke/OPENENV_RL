from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional

from openai import OpenAI

from app.event_engine import EventEngine
from app.models import (
    ActionModel,
    ActionType,
    ApplicationCase,
    DelayedEffect,
    EventType,
    IntakeChannel,
    InternalSubstate,
    ObservationModel,
    PriorityMode,
    QueueSnapshot,
    ServiceType,
    StageType,
)
from app.sector_profiles import get_sector_profile
from app.state_machine import can_advance

if TYPE_CHECKING:
    from app.models import TaskConfig


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

_MODEL_CACHE: dict[tuple[str, str], Any] = {}


# ─────────────────────────────────────────────
# DAY RESULT
# ─────────────────────────────────────────────


class DayResult:
    def __init__(self) -> None:
        self.new_arrivals: int = 0
        self.new_completions: int = 0
        self.new_sla_breaches: int = 0
        self.total_capacity_days: int = 0
        self.idle_officer_days: int = 0
        self.stage_advances: int = 0
        self.newly_unblocked_missing: int = 0
        self.newly_blocked_missing: int = 0
        self.newly_unblocked_enrich: int = 0
        self.field_verif_completed: int = 0
        self.urgent_completed: int = 0
        self.digital_arrivals: int = 0
        self.active_events: list[EventType] = []


# ─────────────────────────────────────────────
# DAY SIMULATOR
# ─────────────────────────────────────────────


class DaySimulator:
    """
    Core daily simulation engine.

    Accepts TWO calling conventions so both env.py and tests work:

    Convention A (tests):
        DaySimulator(task_config=task, rng=rng, event_engine=engine)

    Convention B (env.py legacy):
        DaySimulator(seed=42, task_config=task, sector_registry={})
        — in this case rng and event_engine are built internally.
    """

    def __init__(
        self,
        task_config: "TaskConfig",
        rng: Optional[random.Random] = None,
        event_engine: Optional[EventEngine] = None,
        seed: Optional[int] = None,
        sector_registry: Optional[dict] = None,
    ) -> None:
        self.task_config = task_config
        self.task = task_config

        if rng is not None:
            self.rng = rng
        elif seed is not None:
            self.rng = random.Random(seed)
        else:
            self.rng = random.Random(task_config.seed)

        if event_engine is not None:
            self.event_engine = event_engine
        else:
            _seed = seed if seed is not None else task_config.seed
            self.event_engine = EventEngine(
                seed=_seed,
                scenario_mode=task_config.scenario_mode,
            )

        self.sector_registry = sector_registry or {}
        self.active_cases: list[ApplicationCase] = []
        self.pending_effects: list[DelayedEffect] = []
        self.case_counter: int = 0

    def simulate_day(
        self,
        day: int,
        active_cases: list[ApplicationCase],
        completed_cases: list[ApplicationCase],
        priority_mode: PriorityMode,
        officer_allocations: dict,
    ) -> DayResult:
        result = DayResult()

        events = self.event_engine.get_events_for_day(day, self.task_config)
        params = self.event_engine.apply_events(events, self.task_config)
        result.active_events = list(params.active_events)

        new_cases = self._spawn_arrivals(day, params, result)
        active_cases.extend(new_cases)

        effective_alloc = self._apply_officer_reduction(officer_allocations, params)

        self._resolve_field_verification(day, active_cases, result)
        self._resolve_doc_requests(day, active_cases, result)

        newly_completed: list[ApplicationCase] = []

        for service in self.task_config.enabled_services:
            capacity = effective_alloc.get(service, effective_alloc.get(service.value, 0))
            result.total_capacity_days += int(capacity)

            service_cases = [
                c
                for c in active_cases
                if c.service_type == service and not c.completed and not c.rejected
            ]

            if not service_cases:
                result.idle_officer_days += int(capacity)
                continue

            sorted_cases = self._sort_queue(service_cases, priority_mode)

            for case in sorted_cases:
                if capacity <= 0:
                    break

                from app.state_machine import advance_case

                advanced, final = advance_case(case, day)

                if advanced:
                    capacity -= 1
                    result.stage_advances += 1
                    if final:
                        newly_completed.append(case)
                        if case.is_urgent:
                            result.urgent_completed += 1

        if newly_completed:
            done_ids = {c.case_id for c in newly_completed}
            still_active = [c for c in active_cases if c.case_id not in done_ids]
            active_cases.clear()
            active_cases.extend(still_active)
            completed_cases.extend(newly_completed)
            result.new_completions = len(newly_completed)

        for case in active_cases:
            case.current_day = day
            case.waiting_days += 1
            if day > case.sla_deadline_day and not case.sla_breached:
                case.sla_breached = True
                result.new_sla_breaches += 1

        return result

    def _apply_officer_reduction(self, allocations: dict, params: Any) -> dict:
        reduction = int(getattr(params, "officer_reduction", 0))
        if reduction <= 0:
            return dict(allocations)

        effective = dict(allocations)
        for _ in range(reduction):
            target = max(effective, key=lambda k: effective[k], default=None)
            if target is None or effective[target] <= 0:
                break
            effective[target] -= 1
        return effective

    def _spawn_arrivals(
        self,
        day: int,
        params: Any,
        result: DayResult,
    ) -> list[ApplicationCase]:
        new_cases: list[ApplicationCase] = []

        for service in self.task_config.enabled_services:
            base_rate = self.task_config.arrival_rate_per_day.get(
                service,
                self.task_config.arrival_rate_per_day.get(service.value, 0.0),
            )
            effective_rate = float(base_rate) * float(getattr(params, "arrival_multiplier", 1.0))
            count = int(effective_rate)
            if self.rng.random() < (effective_rate - count):
                count += 1

            for _ in range(count):
                case = self._new_case(service, day, params)
                new_cases.append(case)
                if case.intake_channel == IntakeChannel.DIGITAL:
                    result.digital_arrivals += 1

        result.new_arrivals = len(new_cases)
        return new_cases

    def _new_case(self, service: ServiceType, day: int, params: Any) -> ApplicationCase:
        self.case_counter += 1
        profile = get_sector_profile(service)

        sla_days = int(profile.sla_days * getattr(params, "sla_window_multiplier", 1.0))
        sla_deadline_day = day + sla_days

        digital_ratio = self.task_config.digital_intake_ratio
        channel = (
            IntakeChannel.DIGITAL
            if self.rng.random() < digital_ratio
            else IntakeChannel.PAPER
        )

        base_missing = profile.missing_docs_probability
        override = (self.task_config.missing_docs_probability_override or {}).get(
            service,
            (self.task_config.missing_docs_probability_override or {}).get(service.value),
        )
        if override is not None:
            base_missing = override

        defect_rate = (
            profile.doc_defect_rate_digital
            if channel == IntakeChannel.DIGITAL
            else profile.doc_defect_rate_paper
        )
        eff_missing = min(
            1.0,
            base_missing + getattr(params, "doc_defect_rate_boost", 0.0) * defect_rate,
        )
        has_missing = self.rng.random() < eff_missing

        base_fv = profile.field_verification_probability
        fv_override = (self.task_config.field_verification_probability_override or {}).get(
            service,
            (self.task_config.field_verification_probability_override or {}).get(service.value),
        )
        if fv_override is not None:
            base_fv = fv_override

        eff_fv = min(1.0, base_fv + getattr(params, "field_verification_boost", 0.0))
        has_fv = self.rng.random() < eff_fv
        field_completion_day = day + profile.field_verification_days if has_fv else None

        from app.models import UrgencyProfile

        urgency_profile = profile.urgency_profile
        is_urgent = (
            urgency_profile == UrgencyProfile.HIGH and self.rng.random() < 0.20
        ) or (
            urgency_profile == UrgencyProfile.MODERATE and self.rng.random() < 0.08
        )

        return ApplicationCase(
            case_id=f"case-{self.case_counter:06d}",
            service_type=service,
            arrival_day=day,
            current_day=day,
            sla_deadline_day=sla_deadline_day,
            intake_channel=channel,
            internal_substate=(
                InternalSubstate.BLOCKED_MISSING_DOCS
                if has_missing
                else InternalSubstate.PRE_SCRUTINY
            ),
            public_stage=StageType.SUBMISSION,
            is_urgent=is_urgent,
            has_missing_docs=has_missing,
            field_verification_required=has_fv,
            field_verification_completion_day=field_completion_day,
        )

    def _resolve_field_verification(
        self,
        day: int,
        active_cases: list[ApplicationCase],
        result: DayResult,
    ) -> None:
        for case in active_cases:
            if (
                case.internal_substate == InternalSubstate.FIELD_VERIFICATION_PENDING
                and case.field_verification_completion_day is not None
                and day >= case.field_verification_completion_day
            ):
                case.internal_substate = InternalSubstate.PRE_SCRUTINY
                case.field_verification_completion_day = None
                result.field_verif_completed += 1

    def _resolve_doc_requests(
        self,
        day: int,
        active_cases: list[ApplicationCase],
        result: DayResult,
    ) -> None:
        for case in active_cases:
            if (
                case.internal_substate == InternalSubstate.BLOCKED_MISSING_DOCS
                and case.doc_resolution_day is not None
                and day >= case.doc_resolution_day
            ):
                case.internal_substate = InternalSubstate.PRE_SCRUTINY
                case.doc_resolution_day = None
                result.newly_unblocked_missing += 1

    def _sort_queue(
        self,
        cases: list[ApplicationCase],
        priority_mode: PriorityMode,
    ) -> list[ApplicationCase]:
        eligible = [c for c in cases if can_advance(c)]

        if priority_mode == PriorityMode.URGENT_FIRST:
            return sorted(
                eligible,
                key=lambda c: (not c.is_urgent, -c.sla_risk, c.arrival_day),
            )

        if priority_mode == PriorityMode.OLDEST_FIRST:
            return sorted(eligible, key=lambda c: c.arrival_day)

        if priority_mode == PriorityMode.BACKLOG_CLEARANCE:
            return sorted(
                eligible,
                key=lambda c: (-c.sla_risk, not c.is_urgent, c.arrival_day),
            )

        return sorted(
            eligible,
            key=lambda c: (
                -c.sla_risk if c.sla_risk > 0.8 else 0,
                not c.is_urgent,
                c.arrival_day,
            ),
        )

    def build_queue_snapshot(
        self,
        service: ServiceType,
        active_cases: list[ApplicationCase],
        day: int,
    ) -> QueueSnapshot:
        cases = [
            c
            for c in active_cases
            if c.service_type == service and not c.completed and not c.rejected
        ]

        stage_counts = {s.value: 0 for s in StageType}
        for c in cases:
            stage_counts[c.public_stage.value] = stage_counts.get(c.public_stage.value, 0) + 1

        oldest_age = max((c.waiting_days for c in cases), default=0)
        avg_wait = sum(c.waiting_days for c in cases) / len(cases) if cases else 0.0
        sla_risk = sum(c.sla_risk for c in cases) / len(cases) if cases else 0.0

        return QueueSnapshot(
            service_type=service,
            public_stage_counts=stage_counts,
            total_pending=len(cases),
            total_completed_today=0,
            total_sla_breached=sum(1 for c in cases if c.sla_breached),
            urgent_pending=sum(1 for c in cases if c.is_urgent),
            blocked_missing_docs=sum(
                1
                for c in cases
                if c.internal_substate == InternalSubstate.BLOCKED_MISSING_DOCS
            ),
            field_verification_pending=sum(
                1
                for c in cases
                if c.internal_substate == InternalSubstate.FIELD_VERIFICATION_PENDING
            ),
            oldest_case_age_days=oldest_age,
            avg_waiting_days=round(avg_wait, 2),
            current_sla_risk=round(min(1.0, sla_risk), 3),
        )


# ─────────────────────────────────────────────
# HIGH-LEVEL SIMULATION ORCHESTRATION
# ─────────────────────────────────────────────


class SimulationAgentMode(str, Enum):
    BASELINE_POLICY = "baseline_policy"
    LLM_INFERENCE = "llm_inference"
    TRAINED_RL = "trained_rl"


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
        v = str(value).strip()
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


def _enum_service(value: Any) -> ServiceType | None:
    if value is None or value == "":
        return None
    if isinstance(value, ServiceType):
        return value
    try:
        return ServiceType(str(value))
    except Exception:
        return None


def _enum_priority(value: Any) -> PriorityMode | None:
    if value is None or value == "":
        return None
    if isinstance(value, PriorityMode):
        return value
    try:
        return PriorityMode(str(value))
    except Exception:
        return None


def _action_model_from_kwargs(action_type: ActionType, **kwargs: Any) -> ActionModel:
    service = _enum_service(kwargs.get("service") or kwargs.get("service_target"))
    target_service = _enum_service(kwargs.get("target_service"))
    escalation_target = _enum_service(kwargs.get("escalation_target"))
    priority_mode = _enum_priority(kwargs.get("priority_mode"))
    officer_delta = kwargs.get("officer_delta")
    case_id = kwargs.get("case_id")

    candidates: list[dict[str, Any]] = []

    if action_type == ActionType.ADVANCE_TIME:
        candidates.append({"action_type": action_type})

    elif action_type == ActionType.SET_PRIORITY_MODE:
        candidates.extend(
            [
                {"action_type": action_type, "priority_mode": priority_mode},
            ]
        )

    elif action_type == ActionType.ASSIGN_CAPACITY:
        if service is not None:
            delta = max(1, int(officer_delta or 1))
            candidates.extend(
                [
                    {"action_type": action_type, "service": service, "officer_delta": delta},
                    {"action_type": action_type, "service_target": service, "officer_delta": delta},
                    {
                        "action_type": action_type,
                        "capacity_assignment": {service.value: delta},
                    },
                ]
            )

    elif action_type == ActionType.REQUEST_MISSING_DOCUMENTS:
        if service is not None:
            candidates.extend(
                [
                    {"action_type": action_type, "service": service},
                    {"action_type": action_type, "service_target": service},
                ]
            )

    elif action_type == ActionType.ESCALATE_SERVICE:
        svc = escalation_target or service
        candidates.extend(
            [
                {"action_type": action_type, "service": svc, "case_id": case_id},
                {"action_type": action_type, "service_target": svc, "case_id": case_id},
                {"action_type": action_type, "escalation_target": svc, "case_id": case_id},
            ]
        )

    elif action_type == ActionType.REALLOCATE_OFFICERS:
        if service is not None and target_service is not None:
            delta = max(1, int(officer_delta or 1))
            candidates.extend(
                [
                    {
                        "action_type": action_type,
                        "service": service,
                        "target_service": target_service,
                        "officer_delta": delta,
                    },
                    {
                        "action_type": action_type,
                        "reallocation_delta": {
                            service.value: -delta,
                            target_service.value: delta,
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


def _coerce_action(payload: dict[str, Any] | None) -> ActionModel:
    if not payload:
        return ActionModel(action_type=ActionType.ADVANCE_TIME)

    raw_action_type = payload.get("action_type") or payload.get("actionType")
    try:
        action_type = ActionType(str(raw_action_type))
    except Exception:
        return ActionModel(action_type=ActionType.ADVANCE_TIME)

    service = payload.get("service") or payload.get("service_target") or payload.get("serviceTarget")
    target_service = payload.get("target_service") or payload.get("targetService")
    escalation_target = payload.get("escalation_target") or payload.get("escalationTarget")
    priority_mode = payload.get("priority_mode") or payload.get("priorityMode")
    officer_delta = payload.get("officer_delta") or payload.get("officerDelta")
    case_id = payload.get("case_id") or payload.get("caseId")

    if action_type == ActionType.ASSIGN_CAPACITY and not service:
        assignment = payload.get("capacity_assignment") or {}
        if isinstance(assignment, dict) and assignment:
            service, officer_delta = next(iter(assignment.items()))

    if action_type == ActionType.REALLOCATE_OFFICERS and (not service or not target_service):
        delta_map = payload.get("reallocation_delta") or {}
        if isinstance(delta_map, dict) and len(delta_map) >= 2:
            negatives = [k for k, v in delta_map.items() if int(v) < 0]
            positives = [k for k, v in delta_map.items() if int(v) > 0]
            if negatives and positives:
                service = negatives[0]
                target_service = positives[0]
                officer_delta = abs(int(delta_map[service]))

    return _action_model_from_kwargs(
        action_type,
        service=service,
        target_service=target_service,
        escalation_target=escalation_target,
        priority_mode=priority_mode,
        officer_delta=officer_delta,
        case_id=case_id,
    )


def _recommended_min_steps(task_id: str) -> int:
    if task_id == "cross_department_hard":
        return 70
    if task_id == "mixed_urgency_medium":
        return 60
    return 40


def _queue_snapshot_iter(obs: ObservationModel) -> list[Any]:
    raw = getattr(obs, "queue_snapshots", [])
    if isinstance(raw, dict):
        return list(raw.values())
    if isinstance(raw, list):
        return list(raw)
    try:
        return list(raw)
    except Exception:
        return []


def _queue_service(q: Any) -> ServiceType | None:
    return _enum_service(getattr(q, "service", None) or getattr(q, "service_type", None))


def _queue_active_cases(q: Any) -> int:
    return int(getattr(q, "active_cases", getattr(q, "total_pending", 0)) or 0)


def _queue_missing_docs(q: Any) -> int:
    return int(getattr(q, "missing_docs_cases", getattr(q, "blocked_missing_docs", 0)) or 0)


def _queue_urgent_cases(q: Any) -> int:
    return int(getattr(q, "urgent_cases", getattr(q, "urgent_pending", 0)) or 0)


def _queue_breached_cases(q: Any) -> int:
    return int(getattr(q, "breached_cases", getattr(q, "total_sla_breached", 0)) or 0)


def _queue_avg_age(q: Any) -> float:
    if hasattr(q, "avg_age_days"):
        return float(getattr(q, "avg_age_days") or 0.0)
    if hasattr(q, "oldest_case_age_days"):
        return float(getattr(q, "oldest_case_age_days") or 0.0)
    return float(getattr(q, "avg_waiting_days", 0.0) or 0.0)


def _queue_rows(obs: ObservationModel) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for q in _queue_snapshot_iter(obs):
        service = _queue_service(q)
        if service is None:
            continue
        rows.append(
            {
                "service": service.value,
                "active_cases": _queue_active_cases(q),
                "missing_docs_cases": _queue_missing_docs(q),
                "urgent_cases": _queue_urgent_cases(q),
                "breached_cases": _queue_breached_cases(q),
                "avg_age_days": _queue_avg_age(q),
            }
        )
    return rows


def _pool_allocations(obs: ObservationModel) -> dict[Any, Any]:
    pool = getattr(obs, "officer_pool", None)
    if pool is None:
        return {}
    return getattr(pool, "allocations", getattr(pool, "allocated", {})) or {}


def _reserve_officers(obs: ObservationModel) -> int:
    pool = getattr(obs, "officer_pool", None)
    if pool is None:
        return 0
    for name in ("reserve_officers", "idle_officers", "available_officers"):
        if hasattr(pool, name):
            try:
                return int(getattr(pool, name) or 0)
            except Exception:
                pass
    return 0


def _alloc_for(obs: ObservationModel, service: ServiceType) -> int:
    allocs = _pool_allocations(obs)
    raw = allocs.get(service)
    if raw is None:
        raw = allocs.get(service.value, 0)
    return int(raw or 0)


def _top_backlog_service(
    obs: ObservationModel,
    *,
    exclude: ServiceType | None = None,
) -> ServiceType | None:
    ranked: list[Any] = []
    for q in _queue_snapshot_iter(obs):
        service = _queue_service(q)
        if service is None or service == exclude:
            continue
        ranked.append(q)
    if not ranked:
        return None
    ranked.sort(
        key=lambda q: (
            _queue_active_cases(q) + (2 * _queue_breached_cases(q)) + _queue_urgent_cases(q),
            _queue_avg_age(q),
        ),
        reverse=True,
    )
    return _queue_service(ranked[0])


def _service_with_missing_docs(obs: ObservationModel) -> ServiceType | None:
    candidates = [q for q in _queue_snapshot_iter(obs) if _queue_missing_docs(q) > 0]
    if not candidates:
        return None
    candidates.sort(key=lambda q: (_queue_missing_docs(q), _queue_active_cases(q)), reverse=True)
    return _queue_service(candidates[0])


def _service_with_officers(obs: ObservationModel) -> ServiceType | None:
    services = [s for s in (_queue_service(q) for q in _queue_snapshot_iter(obs)) if s is not None]
    services.sort(key=lambda s: _alloc_for(obs, s), reverse=True)
    for service in services:
        if _alloc_for(obs, service) > 0:
            return service
    return None


def _compute_action_mask(obs: ObservationModel) -> dict[ActionType, bool]:
    has_reserve = _reserve_officers(obs) > 0
    snapshots = _queue_snapshot_iter(obs)
    has_missing = any(_queue_missing_docs(q) > 0 for q in snapshots)
    has_backlog = any(_queue_active_cases(q) > 0 for q in snapshots)
    has_budget = int(getattr(obs, "escalation_budget_remaining", 0) or 0) > 0
    staffed_services = [q for q in snapshots if (_queue_service(q) is not None and _alloc_for(obs, _queue_service(q)) > 0)]
    can_reallocate = len(staffed_services) >= 1 and len(snapshots) >= 2
    return {
        ActionType.SET_PRIORITY_MODE: True,
        ActionType.ADVANCE_TIME: True,
        ActionType.ASSIGN_CAPACITY: has_reserve and has_backlog,
        ActionType.REQUEST_MISSING_DOCUMENTS: has_missing,
        ActionType.ESCALATE_SERVICE: has_budget and has_backlog,
        ActionType.REALLOCATE_OFFICERS: can_reallocate,
    }


def _masked_action_type_hints(obs: ObservationModel) -> tuple[list[str], list[str]]:
    mask = _compute_action_mask(obs)
    allowed = [k.value for k, ok in mask.items() if ok]
    blocked = [k.value for k, ok in mask.items() if not ok]
    return allowed, blocked


def _best_high_impact_action(obs: ObservationModel) -> tuple[ActionModel, str]:
    top_backlog = _top_backlog_service(obs)
    top_missing = _service_with_missing_docs(obs)

    if _reserve_officers(obs) > 0 and top_backlog is not None:
        return (
            _action_model_from_kwargs(
                ActionType.ASSIGN_CAPACITY,
                service=top_backlog,
                officer_delta=1,
            ),
            "high-impact: assign reserve capacity to top backlog service",
        )

    if top_missing is not None:
        return (
            _action_model_from_kwargs(
                ActionType.REQUEST_MISSING_DOCUMENTS,
                service=top_missing,
            ),
            "high-impact: clear missing-document bottleneck",
        )

    if int(getattr(obs, "escalation_budget_remaining", 0) or 0) > 0:
        hot = sorted(
            _queue_snapshot_iter(obs),
            key=lambda q: (_queue_breached_cases(q), _queue_active_cases(q), _queue_urgent_cases(q)),
            reverse=True,
        )
        if hot and (_queue_breached_cases(hot[0]) > 0 or _queue_active_cases(hot[0]) > 0):
            service = _queue_service(hot[0])
            if service is not None:
                return (
                    _action_model_from_kwargs(
                        ActionType.ESCALATE_SERVICE,
                        service=service,
                    ),
                    "high-impact: escalate highest SLA-risk service",
                )

    source = _service_with_officers(obs)
    if source is not None and _alloc_for(obs, source) > 0:
        target = _top_backlog_service(obs, exclude=source)
        if target is not None and target != source:
            return (
                _action_model_from_kwargs(
                    ActionType.REALLOCATE_OFFICERS,
                    service=source,
                    target_service=target,
                    officer_delta=1,
                ),
                "high-impact: reallocate one officer toward highest backlog",
            )

    return ActionModel(action_type=ActionType.ADVANCE_TIME), "fallback: no high-impact action available"


def _repair_action_for_observation(
    action: ActionModel,
    obs: ObservationModel,
) -> tuple[ActionModel, str | None]:
    mask = _compute_action_mask(obs)
    at = action.action_type

    if not bool(mask.get(at, True)):
        fallback, why = _best_high_impact_action(obs)
        return fallback, f"masked {at.value}; {why}"

    if at == ActionType.ADVANCE_TIME:
        return action, None

    if at == ActionType.SET_PRIORITY_MODE:
        if getattr(action, "priority_mode", None) is None:
            return (
                _action_model_from_kwargs(
                    ActionType.SET_PRIORITY_MODE,
                    priority_mode=PriorityMode.BACKLOG_CLEARANCE,
                ),
                "missing priority_mode, defaulted to backlog_clearance",
            )
        return action, None

    if at == ActionType.ASSIGN_CAPACITY:
        reserve = _reserve_officers(obs)
        if reserve <= 0:
            fallback, why = _best_high_impact_action(obs)
            return fallback, f"reserve officers exhausted; {why}"
        service = _enum_service(getattr(action, "service", None) or getattr(action, "service_target", None)) or _top_backlog_service(obs)
        if service is None:
            fallback, why = _best_high_impact_action(obs)
            return fallback, f"no service available for assign_capacity; {why}"
        delta = max(1, int(getattr(action, "officer_delta", 1) or 1))
        delta = min(delta, reserve)
        repaired = _action_model_from_kwargs(
            ActionType.ASSIGN_CAPACITY,
            service=service,
            officer_delta=delta,
        )
        return repaired, "repaired assign_capacity payload"

    if at == ActionType.REQUEST_MISSING_DOCUMENTS:
        service = _enum_service(getattr(action, "service", None) or getattr(action, "service_target", None)) or _service_with_missing_docs(obs)
        if service is None:
            fallback, why = _best_high_impact_action(obs)
            return fallback, f"no missing-doc queue available; {why}"
        repaired = _action_model_from_kwargs(
            ActionType.REQUEST_MISSING_DOCUMENTS,
            service=service,
        )
        return repaired, "repaired request_missing_documents payload"

    if at == ActionType.ESCALATE_SERVICE:
        if int(getattr(obs, "escalation_budget_remaining", 0) or 0) <= 0:
            fallback, why = _best_high_impact_action(obs)
            return fallback, f"escalation budget exhausted; {why}"
        service = (
            _enum_service(getattr(action, "service", None))
            or _enum_service(getattr(action, "service_target", None))
            or _enum_service(getattr(action, "escalation_target", None))
            or _top_backlog_service(obs)
        )
        case_id = getattr(action, "case_id", None)
        if service is None and case_id is None:
            fallback, why = _best_high_impact_action(obs)
            return fallback, f"no escalation target available; {why}"
        repaired = _action_model_from_kwargs(
            ActionType.ESCALATE_SERVICE,
            service=service,
            case_id=case_id,
        )
        return repaired, "repaired escalate_service payload"

    if at == ActionType.REALLOCATE_OFFICERS:
        source = _enum_service(getattr(action, "service", None) or getattr(action, "service_target", None)) or _service_with_officers(obs)
        if source is None:
            fallback, why = _best_high_impact_action(obs)
            return fallback, f"no staffed source service; {why}"
        source_alloc = _alloc_for(obs, source)
        if source_alloc <= 0:
            source = _service_with_officers(obs)
            source_alloc = _alloc_for(obs, source) if source is not None else 0
        if source is None or source_alloc <= 0:
            fallback, why = _best_high_impact_action(obs)
            return fallback, f"insufficient source officers; {why}"

        target = _enum_service(getattr(action, "target_service", None))
        if target is None or target == source:
            target = _top_backlog_service(obs, exclude=source)
        if target is None or target == source:
            fallback, why = _best_high_impact_action(obs)
            return fallback, f"missing distinct target_service; {why}"

        delta = max(1, int(getattr(action, "officer_delta", 1) or 1))
        delta = min(delta, source_alloc)
        repaired = _action_model_from_kwargs(
            ActionType.REALLOCATE_OFFICERS,
            service=source,
            target_service=target,
            officer_delta=delta,
        )
        return repaired, "repaired reallocate_officers payload"

    return action, None


def _model_label_for_mode(agent_mode: SimulationAgentMode) -> str:
    if agent_mode == SimulationAgentMode.BASELINE_POLICY:
        return "baseline_policy"
    if agent_mode == SimulationAgentMode.TRAINED_RL:
        return "trained_rl"
    return os.getenv("MODEL_NAME", "llm_inference")


def _log_step_line(step_row: dict[str, Any]) -> str:
    done = "true" if bool(step_row.get("done")) else "false"
    error = step_row.get("last_action_error") or "null"
    action = json.dumps(step_row.get("action_payload", {}), separators=(",", ":"))
    source = step_row.get("decision_source") or "unknown"
    model = step_row.get("model_used") or "null"
    repair = step_row.get("repair_note") or "null"
    switch_note = step_row.get("switch_note") or "null"
    return (
        f"[STEP] step={step_row.get('step', 0)} action={action} "
        f"reward={float(step_row.get('reward', 0.0)):.2f} done={done} "
        f"error={error} source={source} model={model} repair={repair} switch={switch_note}"
    )


def _resolve_model_path_or_raise(model_path: str) -> str:
    p = Path(model_path).expanduser()
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()

    if p.is_dir():
        candidates = [
            p / "best_model.zip",
            p / "model.zip",
            p / "checkpoint.zip",
        ]
        zip_files = sorted(p.glob("*.zip"))
        candidates.extend(zip_files)
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)

    if p.exists():
        return str(p)

    raise FileNotFoundError(f"Model path not found: {model_path}")


def _load_model_cached_or_raise(model_abs: str, model_type: Literal["maskable", "recurrent"]) -> Any:
    key = (model_abs, model_type)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    if model_type == "recurrent":
        from sb3_contrib import RecurrentPPO

        model = RecurrentPPO.load(model_abs)
    else:
        try:
            from sb3_contrib import MaskablePPO

            model = MaskablePPO.load(model_abs)
        except Exception:
            from stable_baselines3 import PPO

            model = PPO.load(model_abs)

    _MODEL_CACHE[key] = model
    return model


def _safe_invalid_action_count(final_state: Any) -> int:
    if hasattr(final_state, "total_invalid_actions"):
        return int(getattr(final_state, "total_invalid_actions") or 0)
    metrics = getattr(final_state, "metrics", None)
    if metrics is not None and hasattr(metrics, "total_invalid_actions"):
        return int(getattr(metrics, "total_invalid_actions") or 0)
    return 0


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
        recommended = _recommended_min_steps(task_id)
        self.max_steps = max(int(max_steps), int(recommended)) if agent_mode == SimulationAgentMode.LLM_INFERENCE else int(max_steps)
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

        self.env: Any = None
        self.obs: ObservationModel | Any = None
        self.policy: Any = None

        self.rl_env: Any = None
        self.rl_model: Any = None
        self.rl_lstm_state: Any = None
        self.rl_episode_start: Any = None

        self.llm_runtimes: list[dict[str, Any]] = []
        self.llm_route: list[str] = []
        self.llm_model_stats: dict[tuple[str, str], dict[str, Any]] = {}
        self.consecutive_failure_steps = 0
        self.recovery_steps_remaining = 0
        self.auto_switch_count = 0
        self.last_switch_reason: str | None = None

        if self.agent_mode == SimulationAgentMode.TRAINED_RL:
            self._init_trained()
        else:
            self._init_core()

    def start_line(self) -> dict[str, Any]:
        return {
            "log": (
                f"[START] task={self.task_id} env=gov-workflow-openenv "
                f"model={_model_label_for_mode(self.agent_mode)}"
            ),
            "observation": self.obs
        }

    def _init_core(self) -> None:
        from app.baselines import POLICIES, backlog_clearance_policy
        from app.env import GovWorkflowEnv

        self.env = GovWorkflowEnv(task_id=self.task_id)
        self.obs, _ = self.env.reset(seed=self.seed)
        if self.agent_mode == SimulationAgentMode.BASELINE_POLICY:
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
                    clients.append(
                        (
                            OpenAI(base_url=openai_base, api_key=key, timeout=8.0, max_retries=0),
                            f"openai_key_{idx}",
                        )
                    )
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
                    clients.append(
                        (
                            OpenAI(base_url=nvidia_base, api_key=key, timeout=8.0, max_retries=0),
                            f"nvidia_key_{idx}",
                        )
                    )
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
        self.llm_model_stats = {}
        for runtime in runtimes:
            provider = str(runtime.get("provider"))
            for model in runtime.get("models", []):
                self.llm_model_stats[(provider, str(model))] = {
                    "calls": 0,
                    "invalid": 0,
                    "repaired": 0,
                    "failures": 0,
                    "cooldown_until_step": 0,
                }

        openai_runtime = next((rt for rt in runtimes if rt.get("provider") == "openai-compatible"), None)
        nvidia_runtime = next((rt for rt in runtimes if rt.get("provider") == "nvidia"), None)

        openai_route = (
            f"openai-compatible ({len(openai_runtime['clients'])} keys, {len(openai_runtime['models'])} models)"
            if openai_runtime is not None
            else "openai-compatible (unavailable: missing API key/model)"
        )
        nvidia_route = (
            f"nvidia ({len(nvidia_runtime['clients'])} keys, {len(nvidia_runtime['models'])} models)"
            if nvidia_runtime is not None
            else "nvidia (unavailable: missing API key/model)"
        )

        self.llm_route = [
            openai_route,
            nvidia_route,
            "adaptive ranking: prefer models with lower invalid/repaired rates",
            "heuristic fallback (backlog_clearance_policy)",
        ]

    def _rank_runtime_models(self, provider: str, models: list[str]) -> list[str]:
        def _score(model_name: str) -> tuple[float, int]:
            stat = self.llm_model_stats.get((provider, model_name), {})
            calls = max(1, int(stat.get("calls", 0)))
            invalid_rate = float(stat.get("invalid", 0)) / calls
            repaired_rate = float(stat.get("repaired", 0)) / calls
            fail_rate = float(stat.get("failures", 0)) / calls
            cooldown = int(stat.get("cooldown_until_step", 0))
            cooldown_penalty = 1.0 if self.step_idx < cooldown else 0.0
            return (
                invalid_rate * 2.0 + repaired_rate * 1.25 + fail_rate * 1.5 + cooldown_penalty,
                -calls,
            )

        return sorted([str(m) for m in models], key=_score)

    def _llm_action_with_meta(self, obs: ObservationModel) -> tuple[ActionModel, dict[str, Any]]:
        if self.recovery_steps_remaining > 0:
            self.recovery_steps_remaining -= 1
            action, why = _best_high_impact_action(obs)
            return action, {
                "decision_source": "auto_recovery_policy",
                "provider": "heuristic",
                "model_used": "backlog_clearance_policy",
                "llm_attempts": 0,
                "llm_error": None,
                "llm_key_label": None,
                "repair_note": why,
            }

        attempts = 0
        last_error = ""
        allowed_actions, blocked_actions = _masked_action_type_hints(obs)
        schema_hint = {
            "required_fields": {
                "set_priority_mode": ["action_type", "priority_mode"],
                "assign_capacity": ["action_type", "service", "officer_delta"],
                "request_missing_documents": ["action_type", "service"],
                "escalate_service": ["action_type", "service"],
                "advance_time": ["action_type"],
                "reallocate_officers": ["action_type", "service", "target_service", "officer_delta"],
            },
            "allowed_priority_mode": [m.value for m in PriorityMode],
            "allowed_services": [s.value for s in ServiceType],
        }
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
            "5) Always prefer high-impact actions that reduce backlog/SLA risk over no-op loops. "
            "Use lowercase enum values."
        )
        user_prompt = (
            "Observation:\n"
            f"{obs.model_dump_json() if hasattr(obs, 'model_dump_json') else json.dumps(getattr(obs, 'dict', lambda: {})())}\n"
            f"Allowed action types now: {allowed_actions}\n"
            f"Blocked action types now: {blocked_actions}\n"
            f"Action schema hints: {json.dumps(schema_hint, separators=(',', ':'))}\n"
            f"Last action validity: {getattr(obs, 'last_action_valid', True)}\n"
            f"Last action message: {getattr(obs, 'last_action_message', '')}\n"
            "Return action JSON."
        )

        for runtime in self.llm_runtimes:
            provider = str(runtime["provider"])
            ranked_models = self._rank_runtime_models(provider, list(runtime["models"]))
            for client, key_label in runtime["clients"]:
                for model in ranked_models:
                    attempts += 1
                    stat_key = (provider, model)
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
                        if stat_key in self.llm_model_stats:
                            self.llm_model_stats[stat_key]["calls"] += 1
                        return action, {
                            "decision_source": "llm",
                            "provider": provider,
                            "model_used": model,
                            "llm_attempts": attempts,
                            "llm_error": None,
                            "llm_key_label": key_label,
                        }
                    except Exception as exc:
                        last_error = str(exc)
                        stat = self.llm_model_stats.get(stat_key)
                        if stat is not None:
                            stat["calls"] += 1
                            stat["failures"] += 1
                            if stat["failures"] >= 2:
                                stat["cooldown_until_step"] = self.step_idx + 5
                        continue

        action, why = _best_high_impact_action(obs)
        if not self.llm_runtimes:
            last_error = "No LLM credentials configured."
        return action, {
            "decision_source": "heuristic_fallback",
            "provider": "heuristic",
            "model_used": "backlog_clearance_policy",
            "llm_attempts": attempts,
            "llm_error": last_error or None,
            "llm_key_label": None,
            "repair_note": why,
        }

    def _init_trained(self) -> None:
        import numpy as np
        from rl.gov_workflow_env import GovWorkflowGymEnv

        if not self.model_path:
            raise ValueError("model_path is required for trained_rl simulation.")
        model_abs = _resolve_model_path_or_raise(self.model_path)
        self.rl_model = _load_model_cached_or_raise(model_abs, self.model_type)
        self.rl_env = GovWorkflowGymEnv(
            task_id=self.task_id,
            seed=self.seed,
            hard_action_mask=True,
        )
        self.obs, _ = self.rl_env.reset(seed=self.seed)
        self.rl_lstm_state = None
        self.rl_episode_start = np.array([True], dtype=bool)

    def step_once(self) -> tuple[dict[str, Any], str, bool]:
        if self.done:
            raise RuntimeError("Simulation already finished.")

        self.step_idx += 1
        row = self._step_trained() if self.agent_mode == SimulationAgentMode.TRAINED_RL else self._step_core()
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
        return f"[END] success={success} steps={len(self.trace)} score={self.score:.2f} rewards={rewards}"

    def step_line(self, action: dict | ActionModel) -> dict[str, Any]:
        """Test wrapper for executing an action and returning observation + reward."""
        if isinstance(action, dict):
            action = _coerce_action(action)
        self.obs, reward, terminated, truncated, info = self.env.step(action)
        return {"observation": self.obs, "reward": reward}

    def snapshot(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "agent_mode": self.agent_mode.value,
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

        if self.agent_mode == SimulationAgentMode.BASELINE_POLICY:
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
            raw_decision = self.policy(self.obs)
            if isinstance(raw_decision, tuple) and len(raw_decision) == 2:
                action, meta = raw_decision
            else:
                action, meta = raw_decision, {}
            if not isinstance(meta, dict):
                meta = {}

            if not isinstance(action, ActionModel):
                if isinstance(action, dict):
                    action = _coerce_action(action)
                else:
                    action = ActionModel(action_type=ActionType.ADVANCE_TIME)
                    meta["repair_note"] = "non-action output from llm policy, coerced to advance_time"

            allowed_mask = _compute_action_mask(self.obs)
            if not bool(allowed_mask.get(action.action_type, True)):
                masked_fallback, why = _best_high_impact_action(self.obs)
                action = masked_fallback
                if meta.get("decision_source") == "llm":
                    meta["decision_source"] = "llm_repaired"
                meta["repair_note"] = f"action masked at runtime; {why}"

            repaired_action, repair_note = _repair_action_for_observation(action, self.obs)
            if repair_note:
                action = repaired_action
                if meta.get("decision_source") == "llm":
                    meta["decision_source"] = "llm_repaired"
                meta["repair_note"] = repair_note

        self.obs, reward, terminated, truncated, info = self.env.step(action)
        done = bool(terminated or truncated)
        last_action_error = getattr(info, "last_action_error", None)
        if last_action_error is None:
            last_action_error = getattr(info, "action_explanation", None)

        row = {
            "step": self.step_idx,
            "day": self.obs.day,
            "action_type": action.action_type.value,
            "action_payload": action.model_dump(exclude_none=True, mode="json"),
            "reward": float(reward),
            "done": done,
            "backlog": getattr(self.obs, "total_backlog", 0),
            "completed": getattr(self.obs, "total_completed", 0),
            "sla_breaches": getattr(self.obs, "total_sla_breaches", 0),
            "fairness_gap": float(
                getattr(self.obs, "fairness_gap", getattr(self.obs, "fairness_index", 0.0)) or 0.0
            ),
            "escalation_budget_remaining": getattr(self.obs, "escalation_budget_remaining", 0),
            "invalid_action": bool(getattr(info, "invalid_action", False)),
            "last_action_error": last_action_error,
            "queue_rows": _queue_rows(self.obs),
        }
        row.update(meta)

        if self.agent_mode == SimulationAgentMode.LLM_INFERENCE:
            is_repaired = row.get("decision_source") in {"llm_repaired", "auto_recovery_policy"}
            is_invalid = bool(row.get("invalid_action")) or bool(row.get("last_action_error"))
            model_used = str(row.get("model_used") or "")
            provider = str(row.get("provider") or "")
            stat_key = (provider, model_used)
            stat = self.llm_model_stats.get(stat_key)
            if stat is not None:
                if is_repaired:
                    stat["repaired"] += 1
                if is_invalid:
                    stat["invalid"] += 1
                    stat["failures"] += 1
                else:
                    stat["failures"] = max(0, int(stat.get("failures", 0)) - 1)

            is_failure_pattern = is_invalid or is_repaired
            self.consecutive_failure_steps = self.consecutive_failure_steps + 1 if is_failure_pattern else 0

            if self.consecutive_failure_steps >= 4:
                if stat is not None:
                    stat["cooldown_until_step"] = self.step_idx + 6
                self.recovery_steps_remaining = max(self.recovery_steps_remaining, 3)
                self.auto_switch_count += 1
                self.last_switch_reason = "repeated invalid/repaired pattern detected"
                row["switch_note"] = "auto-switched to recovery policy and deprioritized failing model"
                self.consecutive_failure_steps = 0

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

        core_env = self.rl_env.core_env
        core_obs = core_env._build_observation()
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
            "fairness_gap": float(
                getattr(core_obs, "fairness_gap", getattr(core_obs, "fairness_index", 0.0)) or 0.0
            ),
            "escalation_budget_remaining": core_obs.escalation_budget_remaining,
            "invalid_action": bool(info.get("invalid_action", False)),
            "last_action_error": info.get("last_action_error") or info.get("action_explanation"),
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

        from app.graders import grade_episode

        if self.agent_mode == SimulationAgentMode.TRAINED_RL:
            final_state = self.rl_env.core_env.state()
        else:
            final_state = self.env.state()

        gr = grade_episode(final_state)
        self.score = float(gr.score)
        self.grader_name = gr.grader_name

        llm_steps = sum(1 for row in self.trace if row.get("decision_source") in {"llm", "llm_repaired"})
        fallback_steps = sum(
            1 for row in self.trace if row.get("decision_source") in {"heuristic_fallback", "auto_recovery_policy"}
        )
        repaired_steps = sum(
            1 for row in self.trace if row.get("decision_source") in {"llm_repaired", "auto_recovery_policy"}
        )
        total_steps = max(1, len(self.trace))
        invalid_actions = _safe_invalid_action_count(final_state)
        invalid_rate = float(invalid_actions) / float(total_steps)
        repaired_rate = float(repaired_steps) / float(total_steps)

        ranked_models: list[dict[str, Any]] = []
        if self.llm_model_stats:
            for (provider, model), stat in self.llm_model_stats.items():
                calls = int(stat.get("calls", 0))
                if calls <= 0:
                    continue
                ranked_models.append(
                    {
                        "provider": provider,
                        "model": model,
                        "calls": calls,
                        "invalid_rate": float(stat.get("invalid", 0)) / max(1, calls),
                        "repaired_rate": float(stat.get("repaired", 0)) / max(1, calls),
                    }
                )
            ranked_models.sort(key=lambda x: (x["invalid_rate"], x["repaired_rate"], -x["calls"]))

        self.summary = {
            "total_steps": getattr(final_state, "total_steps", len(self.trace)),
            "total_completed": getattr(final_state, "total_completed", 0),
            "total_backlog": getattr(final_state, "total_backlog", 0),
            "total_sla_breaches": getattr(final_state, "total_sla_breaches", 0),
            "fairness_gap": float(getattr(final_state, "fairness_gap", 0.0) or 0.0),
            "total_invalid_actions": invalid_actions,
            "invalid_action_rate": invalid_rate,
            "llm_steps": llm_steps,
            "heuristic_fallback_steps": fallback_steps,
            "llm_repaired_steps": repaired_steps,
            "repaired_action_rate": repaired_rate,
            "auto_switch_count": self.auto_switch_count,
            "last_switch_reason": self.last_switch_reason,
            "effective_max_steps": self.max_steps,
            "recommended_min_steps": _recommended_min_steps(self.task_id),
        }
        if self.agent_mode == SimulationAgentMode.LLM_INFERENCE:
            self.summary["llm_route"] = list(self.llm_route)
            self.summary["llm_model_performance"] = ranked_models
        if self.agent_mode == SimulationAgentMode.TRAINED_RL:
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
    except Exception:
        return ActionModel(action_type=ActionType.ADVANCE_TIME), f"action_{action_idx}"

    row = ACTION_DECODE_TABLE.get(int(action_idx))
    if row is None:
        return ActionModel(action_type=ActionType.ADVANCE_TIME), f"action_{action_idx}"

    action_type, service, priority_mode, delta = row

    try:
        at = ActionType(str(action_type))
    except Exception:
        return ActionModel(action_type=ActionType.ADVANCE_TIME), f"action_{action_idx}"

    if at == ActionType.SET_PRIORITY_MODE:
        action = _action_model_from_kwargs(at, priority_mode=priority_mode)
    elif at == ActionType.ASSIGN_CAPACITY:
        action = _action_model_from_kwargs(at, service=service, officer_delta=delta or 1)
    elif at == ActionType.REQUEST_MISSING_DOCUMENTS:
        action = _action_model_from_kwargs(at, service=service)
    elif at == ActionType.ESCALATE_SERVICE:
        action = _action_model_from_kwargs(at, service=service)
    elif at == ActionType.REALLOCATE_OFFICERS:
        src = _enum_service(service)
        action = (
            _action_model_from_kwargs(at, service=src, target_service=src, officer_delta=delta or 1)
            if src is not None
            else ActionModel(action_type=ActionType.ADVANCE_TIME)
        )
    else:
        action = ActionModel(action_type=ActionType.ADVANCE_TIME)

    return action, at.value