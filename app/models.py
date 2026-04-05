from __future__ import annotations
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class ServiceType(str, Enum):
    PASSPORT = "passport"
    DRIVING_LICENSE = "driving_license"
    GST_REGISTRATION = "gst_registration"
    INCOME_CERTIFICATE = "income_certificate"
    CASTE_CERTIFICATE = "caste_certificate"
    BIRTH_CERTIFICATE = "birth_certificate"
    LAND_REGISTRATION = "land_registration"


class StageType(str, Enum):
    SUBMISSION = "submission"
    DOCUMENT_VERIFICATION = "document_verification"
    FIELD_VERIFICATION = "field_verification"
    APPROVAL = "approval"
    ISSUANCE = "issuance"


class PriorityMode(str, Enum):
    URGENT_FIRST = "urgent_first"
    OLDEST_FIRST = "oldest_first"
    BALANCED = "balanced"
    BACKLOG_CLEARANCE = "backlog_clearance"


class ActionType(str, Enum):
    SET_PRIORITY_MODE = "set_priority_mode"
    ASSIGN_CAPACITY = "assign_capacity"
    REQUEST_MISSING_DOCUMENTS = "request_missing_documents"
    ESCALATE_SERVICE = "escalate_service"
    ADVANCE_TIME = "advance_time"
    REALLOCATE_OFFICERS = "reallocate_officers"


class OfficerPool(BaseModel):
    allocations: dict[ServiceType, int] = Field(default_factory=dict)
    reserve_officers: int = 0

    def total_officers(self) -> int:
        return self.reserve_officers + sum(self.allocations.values())


class QueueSnapshot(BaseModel):
    service: ServiceType
    stage_counts: dict[StageType, int] = Field(default_factory=dict)
    active_cases: int = 0
    missing_docs_cases: int = 0
    escalated_cases: int = 0
    urgent_cases: int = 0
    breached_cases: int = 0
    avg_age_days: float = 0.0


class ServiceCase(BaseModel):
    case_id: str
    service: ServiceType
    stage: StageType = StageType.SUBMISSION
    arrival_day: int
    due_day: int
    urgency: int = Field(ge=1, le=3)
    has_missing_documents: bool = False
    field_verification_required: bool = False
    escalated: bool = False
    total_days: int = 0
    days_in_stage: int = 0
    sla_breached: bool = False
    completed: bool = False


class ObservationModel(BaseModel):
    task_id: str
    day: int
    max_days: int
    priority_mode: PriorityMode
    officer_pool: OfficerPool
    queue_snapshots: list[QueueSnapshot] = Field(default_factory=list)
    total_backlog: int = 0
    total_completed: int = 0
    total_sla_breaches: int = 0
    fairness_gap: float = 0.0
    escalation_budget_remaining: int = 0
    last_action_valid: bool = True
    last_action_message: str = ""


class ActionModel(BaseModel):
    action_type: ActionType
    priority_mode: PriorityMode | None = None
    service: ServiceType | None = None
    target_service: ServiceType | None = None
    case_id: str | None = None
    officer_delta: int = 0
    notes: str | None = None


class RewardModel(BaseModel):
    total_reward: float
    progress_reward: float
    completion_reward: float
    waiting_penalty: float
    sla_penalty: float
    fairness_penalty: float
    invalid_action_penalty: float
    idle_capacity_penalty: float


class StepInfoModel(BaseModel):
    reward_breakdown: RewardModel
    newly_arrived_cases: int = 0
    newly_completed_cases: int = 0
    invalid_action: bool = False
    grader_preview_score: float | None = None
    notes: list[str] = Field(default_factory=list)


class EpisodeMetricsModel(BaseModel):
    total_arrived: int = 0
    total_completed: int = 0
    total_sla_breaches: int = 0
    total_invalid_actions: int = 0
    total_docs_requested: int = 0
    total_docs_cleared: int = 0
    total_urgent_arrived: int = 0
    total_urgent_completed: int = 0
    total_escalations_used: int = 0
    total_wasted_escalations: int = 0
    total_idle_officer_days: int = 0
    total_capacity_days: int = 0


class EpisodeStateModel(BaseModel):
    episode_id: str
    seed: int
    task_id: str
    day: int
    terminated: bool
    truncated: bool
    total_steps: int
    total_completed: int
    total_backlog: int
    total_sla_breaches: int
    action_history_count: int
    fairness_gap: float = 0.0
    escalation_budget_remaining: int = 0
    priority_mode: PriorityMode
    metrics: EpisodeMetricsModel
    action_history: list[dict[str, Any]] = Field(default_factory=list)


class TaskConfig(BaseModel):
    task_id: str
    seed: int
    max_days: int
    arrival_rate: float
    services: list[ServiceType]
    missing_docs_probability: float
    field_verification_probability: float
    escalation_budget: int
    officer_pool: dict[ServiceType, int]
    reserve_officers: int = 0
    initial_cases_by_service: dict[ServiceType, int]
    sla_days: dict[ServiceType, int]
    fairness_threshold: float | None = None


class GraderResult(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    metrics: dict[str, float] = Field(default_factory=dict)
    grader_name: str