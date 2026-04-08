from __future__ import annotations
import random
from uuid import uuid4
from app.config import env_settings
from app.models import (
    ActionModel, ActionType, EpisodeMetricsModel, EpisodeStateModel,
    ObservationModel, OfficerPool, PriorityMode, QueueSnapshot,
    ServiceCase, ServiceType, StageType, StepInfoModel,
)
from app.reward import compute_reward
from app.state_machine import advance_case
from app.tasks import get_task
from app.utils import completion_fairness_gap, priority_key


class GovWorkflowEnv:

    def __init__(self, task_id: str = "district_backlog_easy") -> None:
        self.task_id = task_id
        self.task = get_task(task_id)
        self.rng = random.Random(self.task.seed)
        self.seed = self.task.seed
        self.episode_id = str(uuid4())
        self.case_counter = 0
        self.day = 0
        self.total_steps = 0
        self.terminated = False
        self.truncated = False
        self.priority_mode = PriorityMode.BALANCED
        self.officer_pool = OfficerPool(allocations={}, reserve_officers=0)
        self.active_cases: list[ServiceCase] = []
        self.completed_cases: list[ServiceCase] = []
        self.escalation_budget_remaining = 0
        self.arrived_by_service: dict[ServiceType, int] = {}
        self.completed_by_service: dict[ServiceType, int] = {}
        self.metrics = EpisodeMetricsModel()
        self.action_history: list[dict] = []
        self.last_action_valid = True
        self.last_action_message = ""

    # ── Gymnasium-compatible API ──────────────────────────────────────────────

    def reset(self, seed: int | None = None, options: dict | None = None
              ) -> tuple[ObservationModel, dict]:
        self.task = get_task((options or {}).get("task_id", self.task_id))
        self.task_id = self.task.task_id
        self.seed = self.task.seed if seed is None else seed
        self.rng = random.Random(self.seed)
        self.episode_id = f"{self.task_id}-{self.seed}-{uuid4().hex[:8]}"
        self.case_counter = 0
        self.day = 0
        self.total_steps = 0
        self.terminated = False
        self.truncated = False
        self.priority_mode = PriorityMode.BALANCED
        self.officer_pool = OfficerPool(
            allocations=self.task.officer_pool.copy(),
            reserve_officers=self.task.reserve_officers,
        )
        self.active_cases = []
        self.completed_cases = []
        self.escalation_budget_remaining = self.task.escalation_budget
        self.arrived_by_service = {s: 0 for s in self.task.services}
        self.completed_by_service = {s: 0 for s in self.task.services}
        self.metrics = EpisodeMetricsModel()
        self.action_history = []
        self.last_action_valid = True
        self.last_action_message = "reset"
        self._seed_initial_backlog()
        return self._build_observation(), {"task_id": self.task_id, "seed": self.seed}

    def step(self, action: ActionModel
             ) -> tuple[ObservationModel, float, bool, bool, StepInfoModel]:
        if self.terminated or self.truncated:
            raise RuntimeError("Episode ended. Call reset().")
        self.total_steps += 1
        notes: list[str] = []
        invalid_action = False
        day_metrics = {
            "new_arrivals": 0, "new_completions": 0,
            "stage_advances": 0, "new_sla_breaches": 0, "idle_capacity": 0,
        }
        try:
            action_notes, day_metrics = self._apply_action(action)
            notes.extend(action_notes)
            self.last_action_valid = True
            self.last_action_message = notes[-1] if notes else "ok"
        except ValueError as exc:
            invalid_action = True
            self.metrics.total_invalid_actions += 1
            self.last_action_valid = False
            self.last_action_message = str(exc)
            notes.append(str(exc))

        fairness_gap = completion_fairness_gap(self.arrived_by_service, self.completed_by_service)
        reward = compute_reward(
            stage_advances=day_metrics["stage_advances"],
            completions=day_metrics["new_completions"],
            active_backlog=len(self.active_cases),
            new_sla_breaches=day_metrics["new_sla_breaches"],
            fairness_gap=fairness_gap,
            fairness_threshold=self.task.fairness_threshold,
            invalid_action=invalid_action,
            idle_capacity=day_metrics["idle_capacity"],
        )
        self.terminated = len(self.active_cases) == 0 and self.day > 0
        self.truncated  = self.day >= self.task.max_days and not self.terminated
        preview = None  # grade_episode(self.state()).score # DEPRECATED: Too slow for RL training

        info = StepInfoModel(
            reward_breakdown=reward,
            newly_arrived_cases=day_metrics["new_arrivals"],
            newly_completed_cases=day_metrics["new_completions"],
            invalid_action=invalid_action,
            last_action_error=self.last_action_message if invalid_action else None,
            grader_preview_score=preview,
            notes=notes,
        )
        self.action_history.append({
            "step": self.total_steps, "day": self.day,
            "action": action.model_dump(mode="json"),
            "invalid": invalid_action,
            "message": self.last_action_message,
            "reward": reward.total_reward,
        })
        if self.total_steps >= env_settings.max_steps_per_episode and not self.terminated:
            self.truncated = True

        return self._build_observation(), reward.total_reward, self.terminated, self.truncated, info

    def state(self) -> EpisodeStateModel:
        return EpisodeStateModel(
            episode_id=self.episode_id,
            seed=self.seed,
            task_id=self.task_id,
            day=self.day,
            terminated=self.terminated,
            truncated=self.truncated,
            total_steps=self.total_steps,
            total_completed=len(self.completed_cases),
            total_backlog=len(self.active_cases),
            total_sla_breaches=self.metrics.total_sla_breaches,
            action_history_count=len(self.action_history),
            fairness_gap=completion_fairness_gap(self.arrived_by_service, self.completed_by_service),
            escalation_budget_remaining=self.escalation_budget_remaining,
            priority_mode=self.priority_mode,
            metrics=self.metrics.model_copy(deep=True),
            action_history=list(self.action_history),
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _apply_action(self, action: ActionModel) -> tuple[list[str], dict[str, int]]:
        notes: list[str] = []
        metrics = {"new_arrivals": 0, "new_completions": 0,
                   "stage_advances": 0, "new_sla_breaches": 0, "idle_capacity": 0}

        if action.action_type == ActionType.SET_PRIORITY_MODE:
            if action.priority_mode is None:
                raise ValueError("priority_mode required for set_priority_mode")
            self.priority_mode = action.priority_mode
            notes.append(f"Priority mode set to {action.priority_mode.value}")
            return notes, metrics

        if action.action_type == ActionType.ASSIGN_CAPACITY:
            if action.service is None or action.officer_delta <= 0:
                raise ValueError("service + positive officer_delta required for assign_capacity")
            if action.officer_delta > self.officer_pool.reserve_officers:
                raise ValueError("not enough reserve officers")
            self.officer_pool.allocations[action.service] = (
                self.officer_pool.allocations.get(action.service, 0) + action.officer_delta)
            self.officer_pool.reserve_officers -= action.officer_delta
            notes.append(f"Assigned {action.officer_delta} officers to {action.service.value}")
            return notes, metrics

        if action.action_type == ActionType.REQUEST_MISSING_DOCUMENTS:
            if action.service is None:
                raise ValueError("service required for request_missing_documents")
            candidates = sorted(
                [c for c in self.active_cases
                 if c.service == action.service and c.has_missing_documents],
                key=lambda c: (c.due_day, -c.urgency, c.case_id),
            )
            if not candidates:
                raise ValueError("no missing-document case available")
            case = candidates[0]
            case.has_missing_documents = False
            self.metrics.total_docs_requested += 1
            self.metrics.total_docs_cleared += 1
            notes.append(f"Resolved missing docs for {case.case_id}")
            return notes, metrics

        if action.action_type == ActionType.ESCALATE_SERVICE:
            if self.escalation_budget_remaining <= 0:
                self.metrics.total_wasted_escalations += 1
                raise ValueError("escalation budget exhausted")
            candidates = [c for c in self.active_cases if not c.escalated]
            if action.service:
                candidates = [c for c in candidates if c.service == action.service]
            if action.case_id:
                candidates = [c for c in candidates if c.case_id == action.case_id]
            if not candidates:
                self.metrics.total_wasted_escalations += 1
                raise ValueError("no eligible case to escalate")
            case = sorted(candidates,
                          key=lambda c: (c.due_day, -c.urgency, -c.total_days, c.case_id))[0]
            case.escalated = True
            self.escalation_budget_remaining -= 1
            self.metrics.total_escalations_used += 1
            notes.append(f"Escalated {case.case_id}")
            return notes, metrics

        if action.action_type == ActionType.REALLOCATE_OFFICERS:
            if not action.service or not action.target_service or action.officer_delta <= 0:
                raise ValueError("service, target_service + positive officer_delta required")
            if action.service == action.target_service:
                raise ValueError("source and target services must differ")
            curr = self.officer_pool.allocations.get(action.service, 0)
            if curr < action.officer_delta:
                raise ValueError("not enough officers in source service")
            self.officer_pool.allocations[action.service] = curr - action.officer_delta
            self.officer_pool.allocations[action.target_service] = (
                self.officer_pool.allocations.get(action.target_service, 0) + action.officer_delta)
            notes.append(f"Reallocated {action.officer_delta} officers: "
                         f"{action.service.value} → {action.target_service.value}")
            return notes, metrics

        if action.action_type == ActionType.ADVANCE_TIME:
            metrics = self._advance_one_day()
            notes.append("Advanced simulation by one day")
            return notes, metrics

        raise ValueError(f"Unsupported action_type: {action.action_type}")

    def _advance_one_day(self) -> dict[str, int]:
        metrics = {"new_arrivals": 0, "new_completions": 0,
                   "stage_advances": 0, "new_sla_breaches": 0, "idle_capacity": 0}
        self.day += 1
        metrics["new_arrivals"] = self._spawn_arrivals()
        completed_ids: set[str] = set()
        progressed_ids: set[str] = set()

        for service in self.task.services:
            capacity = self.officer_pool.allocations.get(service, 0)
            self.metrics.total_capacity_days += capacity
            service_cases = [c for c in self.active_cases
                             if c.service == service and not c.completed]
            ordered = sorted(service_cases,
                             key=lambda c: priority_key(c, self.priority_mode, self.day))
            batch = ordered[:capacity]
            idle = max(0, capacity - len(batch))
            metrics["idle_capacity"] += idle
            self.metrics.total_idle_officer_days += idle
            for case in batch:
                progressed, completed = advance_case(case)
                if progressed:
                    metrics["stage_advances"] += 1
                    progressed_ids.add(case.case_id)
                if completed:
                    completed_ids.add(case.case_id)
                    self.completed_by_service[case.service] += 1
                    self.metrics.total_completed += 1
                    metrics["new_completions"] += 1
                    if case.urgency == 3:
                        self.metrics.total_urgent_completed += 1

        if completed_ids:
            still: list[ServiceCase] = []
            for case in self.active_cases:
                (self.completed_cases if case.case_id in completed_ids else still).append(case)
            self.active_cases = still

        for case in self.active_cases:
            case.total_days += 1
            if case.case_id not in progressed_ids:
                case.days_in_stage += 1
            if self.day > case.due_day and not case.sla_breached:
                case.sla_breached = True
                metrics["new_sla_breaches"] += 1
                self.metrics.total_sla_breaches += 1
        return metrics

    def _build_observation(self) -> ObservationModel:
        fairness_gap = completion_fairness_gap(self.arrived_by_service, self.completed_by_service)
        snapshots: list[QueueSnapshot] = []
        for service in self.task.services:
            cases = [c for c in self.active_cases if c.service == service]
            stage_counts = {stage: 0 for stage in StageType}
            for c in cases:
                stage_counts[c.stage] += 1
            avg_age = round(sum(c.total_days for c in cases) / len(cases), 2) if cases else 0.0
            snapshots.append(QueueSnapshot(
                service=service,
                stage_counts=stage_counts,
                active_cases=len(cases),
                missing_docs_cases=sum(1 for c in cases if c.has_missing_documents),
                escalated_cases=sum(1 for c in cases if c.escalated),
                urgent_cases=sum(1 for c in cases if c.urgency == 3),
                breached_cases=sum(1 for c in cases if c.sla_breached),
                avg_age_days=avg_age,
            ))
        return ObservationModel(
            task_id=self.task_id,
            day=self.day,
            max_days=self.task.max_days,
            priority_mode=self.priority_mode,
            officer_pool=self.officer_pool.model_copy(deep=True),
            queue_snapshots=snapshots,
            total_backlog=len(self.active_cases),
            total_completed=len(self.completed_cases),
            total_sla_breaches=self.metrics.total_sla_breaches,
            fairness_gap=fairness_gap,
            escalation_budget_remaining=self.escalation_budget_remaining,
            last_action_valid=self.last_action_valid,
            last_action_message=self.last_action_message,
        )

    def _seed_initial_backlog(self) -> None:
        for service, count in self.task.initial_cases_by_service.items():
            for _ in range(count):
                self.active_cases.append(self._new_case(service))

    def _spawn_arrivals(self) -> int:
        arrivals = int(self.task.arrival_rate)
        if self.rng.random() < (self.task.arrival_rate - arrivals):
            arrivals += 1
        for _ in range(arrivals):
            self.active_cases.append(self._new_case(self.rng.choice(self.task.services)))
        return arrivals

    def _new_case(self, service: ServiceType) -> ServiceCase:
        self.case_counter += 1
        self.arrived_by_service[service] = self.arrived_by_service.get(service, 0) + 1
        self.metrics.total_arrived += 1
        urgency = self.rng.randint(1, 3)
        if urgency == 3:
            self.metrics.total_urgent_arrived += 1
        return ServiceCase(
            case_id=f"case-{self.case_counter:05d}",
            service=service,
            arrival_day=self.day,
            due_day=self.day + self.task.sla_days[service],
            urgency=urgency,
            has_missing_documents=self.rng.random() < self.task.missing_docs_probability,
            field_verification_required=self.rng.random() < self.task.field_verification_probability,
        )
