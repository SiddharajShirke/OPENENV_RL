"""
env.py — Gov Workflow OpenEnv
Gymnasium/OpenEnv-compatible environment aligned with Phase 1 schemas.
"""

from __future__ import annotations

import random
from uuid import uuid4

from app.event_engine import EventEngine
from app.models import (
    ActionModel,
    ActionType,
    ApplicationCase,
    EpisodeStateModel,
    InternalSubstate,
    ObservationModel,
    OfficerPool,
    PriorityMode,
    QueueSnapshot,
    RewardModel,
    ScenarioMode,
    ServiceType,
    StepInfoModel,
    TaskConfig,
)
from app.reward import compute_reward
from app.signal_computer import SignalComputer
from app.engine import DayResult, DaySimulator
from app.tasks import get_task


def completion_fairness_gap(
    arrived_by_service: dict[ServiceType, int],
    completed_by_service: dict[ServiceType, int],
) -> float:
    services = list(arrived_by_service.keys())
    if len(services) < 2:
        return 0.0

    rates = []
    for svc in services:
        arrived = max(1, arrived_by_service.get(svc, 0))
        completed = completed_by_service.get(svc, 0)
        rates.append(completed / arrived)

    return max(rates) - min(rates) if rates else 0.0


class EpisodeMetrics:
    def __init__(self):
        self.total_arrived: int = 0
        self.total_completed: int = 0
        self.total_sla_breaches: int = 0
        self.total_rejected: int = 0
        self.total_invalid_actions: int = 0
        self.total_escalations_used: int = 0
        self.total_wasted_escalations: int = 0
        self.total_docs_requested: int = 0
        self.total_docs_cleared: int = 0
        self.total_idle_officer_days: int = 0
        self.total_capacity_days: int = 0
        self.total_urgent_arrived: int = 0
        self.total_urgent_completed: int = 0
        self.cumulative_reward: float = 0.0

    def to_reward_model(self) -> RewardModel:
        return RewardModel(total_reward=self.cumulative_reward)


class GovWorkflowEnv:
    def __init__(self, task_id: str = "district_backlog_easy", seed: int | None = None) -> None:
        self.task_id = task_id
        self.task: TaskConfig = get_task(task_id)
        self.seed = seed
        self.max_steps_per_episode = max(1, int(self.task.max_days) * 10)
        self._init_episode_state()

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[ObservationModel, dict]:
        task_id = (options or {}).get("task_id", self.task_id)
        self.task = get_task(task_id)
        self.task_id = self.task.task_id

        self.seed = self.task.seed if seed is None else int(seed)
        self.rng = random.Random(self.seed)
        max_steps_override = (options or {}).get("max_steps_per_episode")
        if max_steps_override is None:
            self.max_steps_per_episode = max(1, int(self.task.max_days) * 10)
        else:
            self.max_steps_per_episode = max(1, int(max_steps_override))

        self.episode_id = f"{self.task_id}-s{self.seed}-{uuid4().hex[:6]}"
        self.day = 0
        self.total_steps = 0
        self.terminated = False
        self.truncated = False
        self.priority_mode = PriorityMode.BALANCED

        pool = self.task.initial_officer_pool
        self.officer_pool = OfficerPool(
            total_officers=pool.total_officers,
            available_officers=pool.available_officers,
            allocated=dict(pool.allocated),
            pending_reallocation=dict(getattr(pool, "pending_reallocation", {})),
        )

        self.active_cases: list[ApplicationCase] = []
        self.completed_cases: list[ApplicationCase] = []
        self.escalation_budget_remaining = self.task.escalation_budget

        self.arrived_by_service = {s: 0 for s in self.task.enabled_services}
        self.completed_by_service = {s: 0 for s in self.task.enabled_services}

        self.metrics = EpisodeMetrics()
        self.action_history: list[dict] = []
        self.last_action_valid = True
        self.last_action_message = "reset"
        self.last_action_explanation = ""

        self.event_engine = EventEngine(
            seed=self.seed,
            scenario_mode=self.task.scenario_mode,
        )
        self.simulator = DaySimulator(
            task_config=self.task,
            rng=self.rng,
            event_engine=self.event_engine,
        )
        self.signal_computer = SignalComputer()

        obs = self._build_observation(active_events=[])
        info = {
            "task_id": self.task_id,
            "seed": self.seed,
            "episode_id": self.episode_id,
            "max_days": self.task.max_days,
        }
        return obs, info

    def step(
        self,
        action: ActionModel | dict,
    ) -> tuple[ObservationModel, float, bool, bool, StepInfoModel]:
        if isinstance(action, dict):
            from app.models import ActionModel
            action = ActionModel(**action)
            
        if self.terminated or self.truncated:
            raise RuntimeError("Episode ended — call reset() before stepping.")

        self.total_steps += 1
        invalid_action = False
        day_result = DayResult()

        try:
            notes, day_result = self._apply_action(action, day_result)
            self.last_action_valid = True
            self.last_action_message = notes[-1] if notes else "ok"
            self.last_action_explanation = self.last_action_message
        except ValueError as exc:
            invalid_action = True
            self.metrics.total_invalid_actions += 1
            self.last_action_valid = False
            self.last_action_message = str(exc)
            self.last_action_explanation = f"Invalid: {exc}"

        fairness_gap = completion_fairness_gap(
            self.arrived_by_service,
            self.completed_by_service,
        )

        reward: RewardModel = compute_reward(
            stage_advances=day_result.stage_advances,
            completions=day_result.new_completions,
            active_backlog=len(self.active_cases),
            new_sla_breaches=day_result.new_sla_breaches,
            fairness_gap=fairness_gap,
            fairness_threshold=self.task.fairness_threshold or 0.0,
            invalid_action=invalid_action,
            idle_capacity=day_result.idle_officer_days,
            award_stability_bonus=(action.action_type == ActionType.ADVANCE_TIME),
        )
        self.metrics.cumulative_reward += reward.total_reward

        self.terminated = (
            len(self.active_cases) == 0
            and self.day > 0
            and not invalid_action
        )
        self.truncated = (
            (self.day >= self.task.max_days or self.total_steps >= self.max_steps_per_episode)
            and not self.terminated
        )

        info = StepInfoModel(
            reward_breakdown=reward,
            newly_arrived_cases=day_result.new_arrivals,
            newly_completed_cases=day_result.new_completions,
            newly_sla_breached_cases=day_result.new_sla_breaches,
            newly_resolved_doc_cases=day_result.newly_unblocked_missing,
            invalid_action=invalid_action,
            action_explanation=self.last_action_explanation,
            active_events=day_result.active_events,
            grader_preview_score=0.0,
            effects_resolved_this_step=[],
        )

        self.action_history.append({
            "step": self.total_steps,
            "day": self.day,
            "action": action.model_dump(mode="json"),
            "invalid": invalid_action,
            "message": self.last_action_message,
            "reward": reward.total_reward,
        })

        obs = self._build_observation(active_events=day_result.active_events)
        return obs, reward.total_reward, self.terminated, self.truncated, info

    def count_pending_effects(self) -> int:
        """Count all pending delayed effects waiting to resolve."""
        if hasattr(self, '_pending_effects') and self._pending_effects:
            return len(self._pending_effects)
        if hasattr(self, 'simulator') and hasattr(self.simulator, 'pending_effects'):
            return len(self.simulator.pending_effects)
        if hasattr(self, 'pending_effects'):
            return len(self.pending_effects)
        return 0


    def state(self) -> EpisodeStateModel:
    
        fairness_gap = completion_fairness_gap(
            self.arrived_by_service, self.completed_by_service
        )

        # Compute average waiting days across completed cases
        avg_wait = (
            sum(c.waiting_days for c in self.completed_cases) / len(self.completed_cases)
            if self.completed_cases else 0.0
        )

        return EpisodeStateModel(
            episode_id=self.episode_id,
            task_id=self.task_id,
            seed=self.seed,
            scenario_mode=self.task.scenario_mode,
            day=self.day,
            max_days=self.task.max_days,
            terminated=self.terminated,
            truncated=self.truncated,
            total_steps=self.total_steps,
            total_completed=len(self.completed_cases),
            total_backlog=len(self.active_cases),
            total_sla_breaches=self.metrics.total_sla_breaches,
            total_rejected=self.metrics.total_rejected,
            action_history_count=len(self.action_history),
            cumulative_reward=self.metrics.cumulative_reward,
            officer_pool=self.officer_pool.model_copy(deep=True),
            pending_effects_count=self.count_pending_effects(),
            active_events_today=[],

            # ── Grader-facing fields ──────────────────────────────────
            fairness_gap=round(fairness_gap, 4),
            total_arrived=self.metrics.total_arrived,
            total_docs_requested=self.metrics.total_docs_requested,
            total_docs_cleared=self.metrics.total_docs_cleared,
            total_idle_officer_days=self.metrics.total_idle_officer_days,
            total_capacity_days=self.metrics.total_capacity_days,
            total_urgent_arrived=self.metrics.total_urgent_arrived,
            total_urgent_completed=self.metrics.total_urgent_completed,
            total_escalations_used=self.metrics.total_escalations_used,
            total_wasted_escalations=self.metrics.total_wasted_escalations,
            total_invalid_actions=self.metrics.total_invalid_actions,
            avg_waiting_days=round(avg_wait, 2),

            # Full action log — populated but stripped by API unless requested
            action_history=list(self.action_history),
        )

    def _apply_action(
        self,
        action: ActionModel,
        day_result: DayResult,
    ) -> tuple[list[str], DayResult]:
        notes: list[str] = []

        if action.action_type == ActionType.SET_PRIORITY_MODE:
            if action.priority_mode is None:
                raise ValueError("priority_mode required for set_priority_mode")
            old_mode = self.priority_mode
            self.priority_mode = action.priority_mode
            notes.append(f"Priority mode changed: {old_mode.value} -> {action.priority_mode.value}")
            return notes, day_result

        if action.action_type == ActionType.ASSIGN_CAPACITY:
            cap = action.capacity_assignment
            if not cap:
                raise ValueError("capacity_assignment dict required for assign_capacity")

            for svc_key, delta in cap.items():
                svc = ServiceType(svc_key) if isinstance(svc_key, str) else svc_key
                if svc not in self.task.enabled_services:
                    raise ValueError(f"{svc.value} is not enabled in this task")
                if delta <= 0:
                    raise ValueError("capacity delta must be positive")
                idle = self.officer_pool.idle_officers
                if delta > idle:
                    raise ValueError(f"Only {idle} idle officers available; requested {delta}")
                self.officer_pool.allocated[svc] = self.officer_pool.allocated.get(svc, 0) + delta
                notes.append(f"Assigned {delta} officer(s) to {svc.value}")
            return notes, day_result

        if action.action_type == ActionType.REQUEST_MISSING_DOCUMENTS:
            svc = action.service_target
            if svc is None:
                raise ValueError("service_target required for request_missing_documents")

            candidates = [
                c for c in self.active_cases
                if c.service_type == svc
                and c.internal_substate == InternalSubstate.BLOCKED_MISSING_DOCS
            ]
            if not candidates:
                raise ValueError(f"No BLOCKED_MISSING_DOCS cases for {svc.value}")

            candidates.sort(key=lambda c: (-c.sla_risk, c.arrival_day))
            resolved = 0
            for case in candidates[:3]:
                case.doc_request_sent_day = self.day
                case.doc_resolution_day = self.day + self.rng.randint(2, 3)
                self.metrics.total_docs_requested += 1
                resolved += 1

            notes.append(f"Sent missing-doc requests for {resolved} case(s) in {svc.value}")
            return notes, day_result

        if action.action_type == ActionType.ESCALATE_SERVICE:
            if self.escalation_budget_remaining <= 0:
                self.metrics.total_wasted_escalations += 1
                raise ValueError("Escalation budget exhausted")

            svc = action.escalation_target or action.service_target
            candidates = [
                c for c in self.active_cases
                if (svc is None or c.service_type == svc) and not c.is_urgent
            ]
            if not candidates:
                self.metrics.total_wasted_escalations += 1
                raise ValueError("No eligible non-urgent cases to escalate")

            best = max(candidates, key=lambda c: (c.sla_risk, -c.arrival_day))
            best.is_urgent = True
            self.escalation_budget_remaining -= 1
            self.metrics.total_escalations_used += 1
            notes.append(f"Escalated case {best.case_id} ({best.service_type.value})")
            return notes, day_result

        if action.action_type == ActionType.ADVANCE_TIME:
            day_result = self._advance_one_day()
            notes.append(f"Day {self.day} simulated")
            return notes, day_result

        if action.action_type == ActionType.REALLOCATE_OFFICERS:
            delta = action.reallocation_delta
            if not delta or len(delta) < 2:
                raise ValueError("reallocation_delta must have at least 2 entries")

            total = sum(delta.values())
            if total != 0:
                raise ValueError(f"reallocation_delta must sum to 0 (got {total})")

            for svc_key, change in delta.items():
                svc = ServiceType(svc_key) if isinstance(svc_key, str) else svc_key
                if svc not in self.task.enabled_services:
                    raise ValueError(f"{svc.value} not in enabled services")
                current = self.officer_pool.allocated.get(svc, 0)
                if current + change < 0:
                    raise ValueError(
                        f"Cannot reduce {svc.value} below 0 (current={current}, change={change})"
                    )

            for svc_key, change in delta.items():
                svc = ServiceType(svc_key) if isinstance(svc_key, str) else svc_key
                self.officer_pool.allocated[svc] = self.officer_pool.allocated.get(svc, 0) + change

            changes = ", ".join(f"{k}:{'+' if v > 0 else ''}{v}" for k, v in delta.items())
            notes.append(f"Officers reallocated: {changes}")
            return notes, day_result

        raise ValueError(f"Unsupported action_type: {action.action_type.value}")

    def _advance_one_day(self) -> DayResult:
        self.day += 1

        alloc = dict(self.officer_pool.allocated)
        result = self.simulator.simulate_day(
            day=self.day,
            active_cases=self.active_cases,
            completed_cases=self.completed_cases,
            priority_mode=self.priority_mode,
            officer_allocations=alloc,
        )

        for case in self.completed_cases:
            if getattr(case, "_counted", False):
                continue
            case._counted = True
            svc = case.service_type
            self.completed_by_service[svc] = self.completed_by_service.get(svc, 0) + 1

        for case in self.active_cases:
            if getattr(case, "_arrival_counted", False):
                continue
            case._arrival_counted = True
            svc = case.service_type
            self.arrived_by_service[svc] = self.arrived_by_service.get(svc, 0) + 1
            self.metrics.total_arrived += 1
            if case.is_urgent:
                self.metrics.total_urgent_arrived += 1

        self.metrics.total_completed = len(self.completed_cases)
        self.metrics.total_sla_breaches += result.new_sla_breaches
        self.metrics.total_idle_officer_days += result.idle_officer_days
        self.metrics.total_capacity_days += result.total_capacity_days
        self.metrics.total_urgent_completed += result.urgent_completed
        self.metrics.total_docs_cleared += result.newly_unblocked_missing

        return result

    def _build_observation(self, active_events: list = None) -> ObservationModel:
        active_events = active_events or []

        snapshots: dict[str, QueueSnapshot] = {}
        todays_digital = 0
        todays_arrivals = 0
        today_completed: dict[ServiceType, int] = {}

        for case in self.completed_cases:
            today_completed[case.service_type] = today_completed.get(case.service_type, 0) + 1

        for service in self.task.enabled_services:
            snap = self.simulator.build_queue_snapshot(service, self.active_cases, self.day)
            snap.total_completed_today = today_completed.get(service, 0)
            snapshots[service.value] = snap

        for case in self.active_cases:
            if case.arrival_day == self.day:
                todays_arrivals += 1
                if case.intake_channel.value == "digital":
                    todays_digital += 1

        sigs = self.signal_computer.compute(
            queue_snapshots=snapshots,
            officer_pool=self.officer_pool,
            todays_arrivals=todays_arrivals,
            digital_arrivals=todays_digital,
            capacity_per_day=max(1.0, float(self.officer_pool.available_officers)),
        )

        pending_doc = sum(
            1 for c in self.active_cases
            if c.internal_substate == InternalSubstate.BLOCKED_MISSING_DOCS
            and c.doc_resolution_day is not None
        )
        pending_officer = len(getattr(self.officer_pool, "pending_reallocation", {}))

        return ObservationModel(
            task_id=self.task_id,
            episode_id=self.episode_id,
            day=self.day,
            max_days=self.task.max_days,
            scenario_mode=self.task.scenario_mode,
            officer_pool=self.officer_pool.model_copy(deep=True),
            queue_snapshots=snapshots,
            total_backlog=len(self.active_cases),
            total_completed=len(self.completed_cases),
            total_sla_breaches=self.metrics.total_sla_breaches,
            total_rejected=self.metrics.total_rejected,
            escalation_budget_remaining=self.escalation_budget_remaining,
            backlog_pressure=sigs.backlog_pressure,
            sla_risk_score=sigs.sla_risk_score,
            fairness_index=sigs.fairness_index,
            resource_utilization=sigs.resource_utilization,
            digital_intake_ratio=sigs.digital_intake_ratio,
            blocked_cases_missing_docs=sigs.blocked_cases_missing_docs,
            field_verification_load=sigs.field_verification_load,
            active_events=active_events,
            last_action_valid=self.last_action_valid,
            last_action_message=self.last_action_message,
            last_action_explanation=self.last_action_explanation,
            pending_doc_resolutions=pending_doc,
            pending_officer_reallocations=pending_officer,
        )

    def _init_episode_state(self) -> None:
        self.seed = self.task.seed
        self.rng = random.Random(self.seed)
        self.episode_id = f"{self.task_id}-s{self.seed}-init"
        self.day = 0
        self.total_steps = 0
        self.terminated = False
        self.truncated = False
        self.priority_mode = PriorityMode.BALANCED
        self.officer_pool = OfficerPool(
            total_officers=1,
            available_officers=1,
            allocated={},
            pending_reallocation={},
        )
        self.active_cases: list[ApplicationCase] = []
        self.completed_cases: list[ApplicationCase] = []
        self.escalation_budget_remaining = 0
        self.arrived_by_service: dict[ServiceType, int] = {}
        self.completed_by_service: dict[ServiceType, int] = {}
        self.metrics = EpisodeMetrics()
        self.action_history: list[dict] = []
        self.last_action_valid = True
        self.last_action_message = ""
        self.last_action_explanation = ""
        self.event_engine = EventEngine(seed=self.seed, scenario_mode=ScenarioMode.NORMAL)
        self.simulator = DaySimulator(self.task, self.rng, self.event_engine)
        self.signal_computer = SignalComputer()

    def _count_pending_effects(self) -> int:
        doc_pending = sum(
            1 for c in self.active_cases
            if c.doc_resolution_day is not None
            and c.internal_substate == InternalSubstate.BLOCKED_MISSING_DOCS
        )
        fv_pending = sum(
            1 for c in self.active_cases
            if c.internal_substate == InternalSubstate.FIELD_VERIFICATION_PENDING
            and c.field_verification_completion_day is not None
        )
        return doc_pending + fv_pending

    @property
    def fairness_gap(self) -> float:
        return completion_fairness_gap(self.arrived_by_service, self.completed_by_service)

    @property
    def total_completed(self) -> int:
        return len(self.completed_cases)

    @property
    def total_backlog(self) -> int:
        return len(self.active_cases)
