from __future__ import annotations
import random
from typing import TYPE_CHECKING, List, Optional

from app.event_engine import EventEngine
from app.models import (
    ApplicationCase,
    DelayedEffect,
    EventType,
    IntakeChannel,
    InternalSubstate,
    PriorityMode,
    QueueSnapshot,
    ServiceType,
    StageType,
)
from app.sector_profiles import get_sector_profile
from app.state_machine import can_advance

if TYPE_CHECKING:
    from app.models import TaskConfig


# ─────────────────────────────────────────────
# DAY RESULT
# ─────────────────────────────────────────────

class DayResult:
    def __init__(self):
        self.new_arrivals: int = 0
        self.new_completions: int = 0
        self.new_sla_breaches: int = 0
        self.total_capacity_days: int = 0
        self.idle_officer_days: int = 0
        self.stage_advances: int = 0
        self.newly_unblocked_missing: int = 0
        self.field_verif_completed: int = 0
        self.urgent_completed: int = 0
        self.active_events: List[EventType] = []


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
        # Legacy params kept for env.py compatibility
        seed: Optional[int] = None,
        sector_registry: Optional[dict] = None,
    ):
        self.task_config = task_config
        self.task = task_config          # alias — both names work internally

        # ── RNG ──────────────────────────────────────────────────────────
        if rng is not None:
            self.rng = rng
        elif seed is not None:
            self.rng = random.Random(seed)
        else:
            self.rng = random.Random(task_config.seed)

        # ── EVENT ENGINE ─────────────────────────────────────────────────
        if event_engine is not None:
            self.event_engine = event_engine
        else:
            _seed = seed if seed is not None else task_config.seed
            self.event_engine = EventEngine(
                seed=_seed,
                scenario_mode=task_config.scenario_mode,
            )

        # ── SECTOR REGISTRY ──────────────────────────────────────────────
        self.sector_registry = sector_registry or {}

        # ── STATE ────────────────────────────────────────────────────────
        self.active_cases: List[ApplicationCase] = []
        self.pending_effects: List[DelayedEffect] = []
        self.case_counter: int = 0          # total cases ever created (unique IDs)


    # ─────────────────────────────────────────────
    # MAIN SIMULATE DAY
    # ─────────────────────────────────────────────

    def simulate_day(
        self,
        day: int,
        active_cases: List[ApplicationCase],
        completed_cases: List[ApplicationCase],
        priority_mode: PriorityMode,
        officer_allocations: dict,
    ) -> DayResult:
        result = DayResult()

        # 1. Resolve today's events
        events = self.event_engine.get_events_for_day(day, self.task_config)
        params = self.event_engine.apply_events(events, self.task_config)
        result.active_events = list(params.active_events)

        # 2. Spawn new arrivals
        new_cases = self._spawn_arrivals(day, params, result)
        active_cases.extend(new_cases)

        # 3. Apply officer reduction from events
        effective_alloc = self._apply_officer_reduction(officer_allocations, params)

        # 4. Resolve time-based pending items
        self._resolve_field_verification(day, active_cases, result)
        self._resolve_doc_requests(day, active_cases, result)

        # 5. Process each service queue
        newly_completed: List[ApplicationCase] = []

        for service in self.task_config.enabled_services:
            capacity = effective_alloc.get(
                service,
                effective_alloc.get(service.value, 0)
            )
            result.total_capacity_days += capacity

            service_cases = [
                c for c in active_cases
                if c.service_type == service and not c.completed and not c.rejected
            ]

            if not service_cases:
                result.idle_officer_days += capacity
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

        # 6. Remove completed cases from active list
        if newly_completed:
            done_ids = {c.case_id for c in newly_completed}
            still_active = [c for c in active_cases if c.case_id not in done_ids]
            active_cases.clear()
            active_cases.extend(still_active)
            completed_cases.extend(newly_completed)
            result.new_completions = len(newly_completed)

        # 7. Age all remaining cases — use sla_deadline_day (correct field name)
        for case in active_cases:
            case.current_day = day
            case.waiting_days += 1
            if day > case.sla_deadline_day and not case.sla_breached:
                case.sla_breached = True
                result.new_sla_breaches += 1

        return result


    # ─────────────────────────────────────────────
    # INTERNAL HELPERS
    # ─────────────────────────────────────────────

    def _apply_officer_reduction(self, allocations: dict, params) -> dict:
        reduction = getattr(params, "officer_reduction", 0)
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
        self, day: int, params, result: DayResult
    ) -> List[ApplicationCase]:
        new_cases: List[ApplicationCase] = []

        for service in self.task_config.enabled_services:
            base_rate = self.task_config.arrival_rate_per_day.get(
                service,
                self.task_config.arrival_rate_per_day.get(service.value, 0.0),
            )
            effective_rate = base_rate * getattr(params, "arrival_multiplier", 1.0)
            count = int(effective_rate)
            if self.rng.random() < (effective_rate - count):
                count += 1

            for _ in range(count):
                new_cases.append(self._new_case(service, day, params))

        result.new_arrivals = len(new_cases)
        return new_cases

    def _new_case(self, service: ServiceType, day: int, params) -> ApplicationCase:
        self.case_counter += 1
        profile = get_sector_profile(service)

        # ── SLA deadline — use sla_deadline_day (matches models.py field name) ──
        sla_days = int(profile.sla_days * getattr(params, "sla_window_multiplier", 1.0))
        sla_deadline_day = day + sla_days        # FIXED: was "due_day"

        # ── Intake channel ────────────────────────────────────────────────
        digital_ratio = self.task_config.digital_intake_ratio
        channel = (
            IntakeChannel.DIGITAL
            if self.rng.random() < digital_ratio
            else IntakeChannel.PAPER
        )

        # ── Missing docs probability ──────────────────────────────────────
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
            base_missing + getattr(params, "doc_defect_rate_boost", 0.0) * defect_rate
        )
        has_missing = self.rng.random() < eff_missing

        # ── Field verification probability ────────────────────────────────
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

        # ── Urgency ───────────────────────────────────────────────────────
        from app.models import UrgencyProfile
        urgency_profile = profile.urgency_profile
        is_urgent = (
            urgency_profile == UrgencyProfile.HIGH
            and self.rng.random() < 0.20
        ) or (
            urgency_profile == UrgencyProfile.MODERATE
            and self.rng.random() < 0.08
        )

        # ── Build ApplicationCase using correct field names from models.py ──
        return ApplicationCase(
            case_id=f"case-{self.case_counter:06d}",
            service_type=service,
            arrival_day=day,
            current_day=day,
            sla_deadline_day=sla_deadline_day,          # FIXED: was due_day
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
        active_cases: List[ApplicationCase],
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
        active_cases: List[ApplicationCase],
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
        cases: List[ApplicationCase],
        priority_mode: PriorityMode,
    ) -> List[ApplicationCase]:
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

        # BALANCED (default)
        return sorted(
            eligible,
            key=lambda c: (
                -c.sla_risk if c.sla_risk > 0.8 else 0,
                not c.is_urgent,
                c.arrival_day,
            ),
        )


    # ─────────────────────────────────────────────
    # QUEUE SNAPSHOT
    # ─────────────────────────────────────────────

    def build_queue_snapshot(
        self,
        service: ServiceType,
        active_cases: List[ApplicationCase],
        day: int,
    ) -> QueueSnapshot:
        cases = [
            c for c in active_cases
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
                1 for c in cases
                if c.internal_substate == InternalSubstate.BLOCKED_MISSING_DOCS
            ),
            field_verification_pending=sum(
                1 for c in cases
                if c.internal_substate == InternalSubstate.FIELD_VERIFICATION_PENDING
            ),
            oldest_case_age_days=oldest_age,
            avg_waiting_days=round(avg_wait, 2),
            current_sla_risk=round(min(1.0, sla_risk), 3),
        )


