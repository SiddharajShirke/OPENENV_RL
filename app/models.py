"""
models.py — Gov Workflow OpenEnv v2.0 — Phase 2 FULL FILE
Adds: DocEnrichmentType, doc_enrichment fields on ApplicationCase,
      blocked_cases_enrichment / pending_enrichment_lookups on observation,
      INTERNAL_TO_PUBLIC_STAGE mapping,
      SectorProfile enrichment fields.
"""

from __future__ import annotations
from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
import uuid


# ─────────────────────────────────────────────
# ENUMS
# ─────────────────────────────────────────────

class ServiceType(str, Enum):
    PASSPORT            = "passport"
    DRIVING_LICENSE     = "driving_license"
    GST_REGISTRATION    = "gst_registration"
    INCOME_CERTIFICATE  = "income_certificate"
    CASTE_CERTIFICATE   = "caste_certificate"
    BIRTH_CERTIFICATE   = "birth_certificate"
    LAND_REGISTRATION   = "land_registration"


class StageType(str, Enum):
    SUBMISSION            = "submission"
    DOCUMENT_VERIFICATION = "document_verification"
    FIELD_VERIFICATION    = "field_verification"
    APPROVAL              = "approval"
    ISSUANCE              = "issuance"


class InternalSubstate(str, Enum):
    PRE_SCRUTINY                 = "pre_scrutiny"
    DOC_VALIDATION               = "doc_validation"
    SERVICE_SPECIFIC_VALIDATION  = "service_specific_validation"
    FIELD_VERIFICATION_PENDING   = "field_verification_pending"
    DECISION_PENDING             = "decision_pending"
    ISSUANCE_READY               = "issuance_ready"
    BLOCKED_MISSING_DOCS         = "blocked_missing_docs"
    BLOCKED_ENRICHMENT           = "blocked_enrichment"
    COMPLETED                    = "completed"
    REJECTED                     = "rejected"


# ── Phase 2 addition ──────────────────────────────────────────────────────────
class DocEnrichmentType(str, Enum):
    """External lookup needed for document verification."""
    NONE                  = "none"
    PAST_LAND_RECORDS     = "past_land_records"       # Land Registration — Revenue DB
    FAMILY_CASTE_HISTORY  = "family_caste_history"    # Caste Certificate — Caste Registry
    POLICE_VERIFICATION   = "police_verification"     # Passport — Police Station
    TAX_RECORD_CROSS_CHECK= "tax_record_cross_check"  # GST Registration — Tax DB


# Public stage mapping — used by state_machine.build_public_stage
INTERNAL_TO_PUBLIC_STAGE: dict = {
    "pre_scrutiny":                  "submission",
    "doc_validation":                "document_verification",
    "service_specific_validation":   "document_verification",
    "field_verification_pending":    "field_verification",
    "decision_pending":              "approval",
    "issuance_ready":                "issuance",
    "blocked_missing_docs":          "document_verification",
    "blocked_enrichment":            "document_verification",
    "completed":                     "issuance",
    "rejected":                      "approval",
}


class PriorityMode(str, Enum):
    URGENT_FIRST       = "urgent_first"
    OLDEST_FIRST       = "oldest_first"
    BALANCED           = "balanced"
    BACKLOG_CLEARANCE  = "backlog_clearance"


class ActionType(str, Enum):
    SET_PRIORITY_MODE         = "set_priority_mode"
    ASSIGN_CAPACITY           = "assign_capacity"
    REQUEST_MISSING_DOCUMENTS = "request_missing_documents"
    ESCALATE_SERVICE          = "escalate_service"
    ADVANCE_TIME              = "advance_time"
    REALLOCATE_OFFICERS       = "reallocate_officers"


class EventType(str, Enum):
    SURGE_APPLICATIONS        = "surge_applications"
    OFFICER_UNAVAILABLE       = "officer_unavailable"
    DOCUMENT_REJECTION_SPIKE  = "document_rejection_spike"
    REVENUE_DB_DELAY          = "revenue_db_delay"
    SLA_ESCALATION_ORDER      = "sla_escalation_order"
    NO_EVENT                  = "no_event"


class ScenarioMode(str, Enum):
    NORMAL           = "normal"
    CRISIS           = "crisis"
    EXTREME_OVERLOAD = "extreme_overload"


class UrgencyProfile(str, Enum):
    LOW             = "low"
    MODERATE        = "moderate"
    HIGH            = "high"
    LOW_BUT_STICKY  = "low_but_sticky"


class IntakeChannel(str, Enum):
    DIGITAL = "digital"
    PAPER   = "paper"
    HYBRID  = "hybrid"


class DelayedEffectType(str, Enum):
    DOC_REQUEST_RESOLUTION = "doc_request_resolution"
    OFFICER_REALLOCATION   = "officer_reallocation"
    ESCALATION_RELIEF      = "escalation_relief"


# ─────────────────────────────────────────────
# SECTOR / SERVICE CONFIGURATION
# ─────────────────────────────────────────────

class SectorProfile(BaseModel):
    service_type:                   ServiceType
    sector_name:                    str
    missing_docs_probability:       float = Field(ge=0.0, le=1.0)
    doc_defect_rate_digital:        float = Field(ge=0.0, le=1.0)
    doc_defect_rate_paper:          float = Field(ge=0.0, le=1.0)
    field_verification_probability: float = Field(ge=0.0, le=1.0)
    manual_scrutiny_intensity:      float = Field(ge=0.0, le=1.0)
    decision_backlog_sensitivity:   float = Field(ge=0.0, le=1.0)
    system_dependency_risk:         float = Field(ge=0.0, le=1.0)
    sla_days:                       int   = Field(ge=1)
    urgency_profile:                UrgencyProfile
    base_processing_rate:           float = Field(ge=0.1)
    field_verification_days:        int   = Field(ge=1)
    # ── Phase 2: enrichment ─────────────────────────────────────────
    doc_enrichment_type:                DocEnrichmentType  = DocEnrichmentType.NONE
    doc_enrichment_probability:         float              = Field(default=0.0, ge=0.0, le=1.0)
    doc_enrichment_delay_days_min:      int                = Field(default=1, ge=1)
    doc_enrichment_delay_days_max:      int                = Field(default=3, ge=1)


class OfficerPool(BaseModel):
    total_officers:       int           = Field(ge=1)
    available_officers:   int           = Field(ge=0)
    allocated:            Dict[str, int] = Field(default_factory=dict)
    pending_reallocation: Dict[str, int] = Field(default_factory=dict)

    @property
    def idle_officers(self) -> int:
        return self.available_officers - sum(self.allocated.values())


# ─────────────────────────────────────────────
# CASE MODEL  (Phase 2: enrichment fields added)
# ─────────────────────────────────────────────

class ApplicationCase(BaseModel):
    case_id:              str               = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    service_type:         ServiceType
    internal_substate:    InternalSubstate  = InternalSubstate.PRE_SCRUTINY
    public_stage:         StageType         = StageType.SUBMISSION

    arrival_day:          int               = Field(ge=0)
    current_day:          int               = Field(ge=0)
    sla_deadline_day:     int               = Field(ge=0)
    days_in_current_stage:int               = Field(default=0, ge=0)
    waiting_days:         int               = Field(default=0, ge=0)

    is_urgent:            bool              = False
    intake_channel:       IntakeChannel     = IntakeChannel.DIGITAL
    has_missing_docs:     bool              = False
    doc_request_sent_day: Optional[int]     = None
    doc_resolution_day:   Optional[int]     = None
    field_verification_required:          bool           = False
    field_verification_completion_day:    Optional[int]  = None

    sla_breached:         bool              = False
    completed:            bool              = False
    rejected:             bool              = False

    # ── Phase 2: enrichment ─────────────────────────────────────────
    doc_enrichment_type:     DocEnrichmentType  = DocEnrichmentType.NONE
    doc_enrichment_triggered:bool               = False
    enrichment_resolution_day:Optional[int]     = None
    doc_enrichment_reason:   Optional[str]      = None

    @property
    def days_until_sla(self) -> int:
        return max(0, self.sla_deadline_day - self.current_day)

    @property
    def sla_risk(self) -> float:
        total_window = self.sla_deadline_day - self.arrival_day
        if total_window <= 0:
            return 1.0
        elapsed = self.current_day - self.arrival_day
        return min(1.0, elapsed / total_window)


class QueueSnapshot(BaseModel):
    service_type:              ServiceType
    public_stage_counts:       Dict[str, int] = Field(default_factory=dict)
    total_pending:             int            = Field(default=0, ge=0)
    total_completed_today:     int            = Field(default=0, ge=0)
    total_sla_breached:        int            = Field(default=0, ge=0)
    urgent_pending:            int            = Field(default=0, ge=0)
    blocked_missing_docs:      int            = Field(default=0, ge=0)
    blocked_enrichment:        int            = Field(default=0, ge=0)   # Phase 2
    field_verification_pending:int            = Field(default=0, ge=0)
    oldest_case_age_days:      int            = Field(default=0, ge=0)
    avg_waiting_days:          float          = Field(default=0.0, ge=0.0)
    current_sla_risk:          float          = Field(default=0.0, ge=0.0, le=1.0)


# ─────────────────────────────────────────────
# DELAYED EFFECT MODEL
# ─────────────────────────────────────────────

class DelayedEffect(BaseModel):
    effect_id:       str                    = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    effect_type:     DelayedEffectType
    target_service:  Optional[ServiceType]  = None
    target_case_id:  Optional[str]          = None
    resolution_day:  int                    = Field(ge=0)
    magnitude:       float                  = Field(default=1.0)
    description:     str                    = Field(default="")


# ─────────────────────────────────────────────
# OBSERVATION MODEL  (Phase 2: enrichment signals added)
# ─────────────────────────────────────────────

class ObservationModel(BaseModel):
    task_id:         str
    episode_id:      str
    day:             int                    = Field(ge=0)
    max_days:        int                    = Field(ge=1)
    scenario_mode:   ScenarioMode           = ScenarioMode.NORMAL
    officer_pool:    OfficerPool
    queue_snapshots: Dict[str, QueueSnapshot] = Field(default_factory=dict)

    total_backlog:             int          = Field(default=0, ge=0)
    total_completed:           int          = Field(default=0, ge=0)
    total_sla_breaches:        int          = Field(default=0, ge=0)
    total_rejected:            int          = Field(default=0, ge=0)
    escalation_budget_remaining:int         = Field(default=0, ge=0)

    # Compressed signals
    backlog_pressure:          float        = Field(default=0.0, ge=0.0, le=1.0)
    sla_risk_score:            float        = Field(default=0.0, ge=0.0, le=1.0)
    fairness_index:            float        = Field(default=1.0, ge=0.0, le=1.0)
    resource_utilization:      float        = Field(default=0.0, ge=0.0, le=1.0)
    digital_intake_ratio:      float        = Field(default=0.5, ge=0.0, le=1.0)
    blocked_cases_missing_docs:int          = Field(default=0, ge=0)
    blocked_cases_enrichment:  int          = Field(default=0, ge=0)   # Phase 2
    field_verification_load:   float        = Field(default=0.0, ge=0.0, le=1.0)

    active_events:             List[EventType] = Field(default_factory=list)

    last_action_valid:         bool         = True
    last_action_message:       str          = ""
    last_action_explanation:   str          = Field(default="")

    pending_doc_resolutions:   int          = Field(default=0, ge=0)
    pending_enrichment_lookups:int          = Field(default=0, ge=0)  # Phase 2
    pending_officer_reallocations:int       = Field(default=0, ge=0)


# ─────────────────────────────────────────────
# ACTION / REWARD / STATE MODELS (unchanged)
# ─────────────────────────────────────────────

class ActionModel(BaseModel):
    action_type:          ActionType
    service_target:       Optional[ServiceType]  = None
    priority_mode:        Optional[PriorityMode] = None
    reallocation_delta:   Optional[Dict[str, int]] = None
    escalation_target:    Optional[ServiceType]  = None
    capacity_assignment:  Optional[Dict[str, int]] = None
    notes:                Optional[str]           = None


class RewardModel(BaseModel):
    total_reward:              float = 0.0
    progress_reward:           float = 0.0
    completion_reward:         float = 0.0
    recovery_reward:           float = 0.0
    stability_bonus:           float = 0.0
    waiting_penalty:           float = 0.0
    sla_penalty:               float = 0.0
    fairness_penalty:          float = 0.0
    invalid_action_penalty:    float = 0.0
    idle_capacity_penalty:     float = 0.0
    oscillation_penalty:       float = 0.0


class EpisodeStateModel(BaseModel):
    """Internal episode state exposed via GET /state and POST /state endpoints."""
    episode_id: str
    task_id: str
    seed: int
    scenario_mode: ScenarioMode
    day: int = Field(ge=0)
    max_days: int = Field(ge=1)
    terminated: bool = False
    truncated: bool = False
    total_steps: int = Field(default=0, ge=0)
    total_completed: int = Field(default=0, ge=0)
    total_backlog: int = Field(default=0, ge=0)
    total_sla_breaches: int = Field(default=0, ge=0)
    total_rejected: int = Field(default=0, ge=0)
    action_history_count: int = Field(default=0, ge=0)
    cumulative_reward: float = 0.0
    cumulative_reward_breakdown: RewardModel = Field(default_factory=RewardModel)
    officer_pool: Optional[OfficerPool] = None
    pending_effects_count: int = Field(default=0, ge=0)
    active_events_today: List[EventType] = Field(default_factory=list)

    # ── Grader-facing fields ──────────────────────────────────────
    # These are populated by env.state() so graders never need to
    # reach into private EpisodeMetrics.
    fairness_gap: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Cross-service completion fairness gap at episode end"
    )
    total_arrived: int = Field(
        default=0, ge=0,
        description="Total cases that arrived across all services"
    )
    total_docs_requested: int = Field(
        default=0, ge=0,
        description="Total missing-doc requests sent"
    )
    total_docs_cleared: int = Field(
        default=0, ge=0,
        description="Total missing-doc cases subsequently resolved"
    )
    total_idle_officer_days: int = Field(
        default=0, ge=0,
        description="Cumulative officer-days wasted idle"
    )
    total_capacity_days: int = Field(
        default=0, ge=0,
        description="Cumulative total officer-days available"
    )
    total_urgent_arrived: int = Field(
        default=0, ge=0,
        description="Total urgent cases that arrived"
    )
    total_urgent_completed: int = Field(
        default=0, ge=0,
        description="Total urgent cases completed"
    )
    total_escalations_used: int = Field(
        default=0, ge=0,
        description="Total escalation actions consumed"
    )
    total_wasted_escalations: int = Field(
        default=0, ge=0,
        description="Escalations used on already-urgent or ineligible cases"
    )
    total_invalid_actions: int = Field(
        default=0, ge=0,
        description="Total invalid actions submitted by agent"
    )
    avg_waiting_days: float = Field(
        default=0.0, ge=0.0,
        description="Mean waiting days across all completed cases"
    )

    # ── Full action log (optional, stripped by default) ──────────
    action_history: Optional[List[dict]] = Field(
        default=None,
        description="Step-by-step action log. Stripped in normal API responses."
    )


class StepInfoModel(BaseModel):
    reward_breakdown:              RewardModel  = Field(default_factory=RewardModel)
    newly_arrived_cases:           int          = Field(default=0, ge=0)
    newly_completed_cases:         int          = Field(default=0, ge=0)
    newly_sla_breached_cases:      int          = Field(default=0, ge=0)
    newly_resolved_doc_cases:      int          = Field(default=0, ge=0)
    invalid_action:                bool         = False
    action_explanation:            str          = ""
    active_events:                 List[EventType] = Field(default_factory=list)
    grader_preview_score:          float        = Field(default=0.0, ge=0.0, le=1.0)
    effects_resolved_this_step:    List[str]    = Field(default_factory=list)


class TaskConfig(BaseModel):
    task_id:                str
    display_name:           str
    difficulty:             str
    scenario_mode:          ScenarioMode
    seed:                   int
    max_days:               int                    = Field(ge=1)
    enabled_services:       List[ServiceType]
    arrival_rate_per_day:   Dict[str, float]
    digital_intake_ratio:   float                  = Field(default=0.6, ge=0.0, le=1.0)
    initial_officer_pool:   OfficerPool
    missing_docs_probability_override:       Optional[Dict[str, float]] = None
    field_verification_probability_override: Optional[Dict[str, float]] = None
    escalation_budget:      int                    = Field(ge=0)
    fairness_threshold:     Optional[float]        = Field(default=None, ge=0.0, le=1.0)
    event_probability:      float                  = Field(default=0.1, ge=0.0, le=1.0)
    allowed_events:         List[EventType]        = Field(default_factory=list)


class GraderResult(BaseModel):
    """
    Final deterministic score for a completed or in-progress episode.
    Range: [0.0, 1.0].

    Design decision: exposes .score and .grader_name as convenience aliases,
    plus a .metrics dict for easy serialization to JSON by main.py endpoints.
    The named fields (completion_rate, sla_compliance_rate, etc.) remain
    for typed access in tests and baselines.
    """
    task_id: str = ""
    episode_id: str = ""
    grader_name: str = ""                          # "easy" | "medium" | "hard"

    # Primary scalar — use result.score everywhere
    score: float = Field(default=0.0, ge=0.0, le=1.0)

    # Named metric components
    completion_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    sla_compliance_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    idle_efficiency: float = Field(default=1.0, ge=0.0, le=1.0)
    document_rework_quality: float = Field(default=1.0, ge=0.0, le=1.0)
    urgent_served_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    fairness_score: float = Field(default=1.0, ge=0.0, le=1.0)
    escalation_discipline: float = Field(default=1.0, ge=0.0, le=1.0)
    fairness_gap: float = Field(default=0.0, ge=0.0, le=1.0)

    # Episode counters — populated from EpisodeStateModel
    total_cases_arrived: int = 0
    total_completed: int = 0
    total_sla_breached: int = 0
    total_rejected: int = 0
    avg_waiting_days: float = 0.0

    @property
    def metrics(self) -> dict:
        """
        Convenience dict for JSON serialization in API endpoints.
        main.py uses result.metrics directly in GradeResponse.
        """
        return {
            "completion_rate":         round(self.completion_rate, 4),
            "sla_compliance_rate":     round(self.sla_compliance_rate, 4),
            "idle_efficiency":         round(self.idle_efficiency, 4),
            "document_rework_quality": round(self.document_rework_quality, 4),
            "urgent_served_rate":      round(self.urgent_served_rate, 4),
            "fairness_score":          round(self.fairness_score, 4),
            "escalation_discipline":   round(self.escalation_discipline, 4),
            "fairness_gap":            round(self.fairness_gap, 4),
            "total_cases_arrived":     self.total_cases_arrived,
            "total_completed":         self.total_completed,
            "total_sla_breached":      self.total_sla_breached,
            "total_rejected":          self.total_rejected,
            "avg_waiting_days":        round(self.avg_waiting_days, 2),
        }


class ResetRequest(BaseModel):
    task_id:        str
    seed:           Optional[int]           = None
    scenario_mode:  Optional[ScenarioMode]  = None


class ResetResponse(BaseModel):
    observation:    ObservationModel
    info:           dict
    episode_id:     str


class StepRequest(BaseModel):
    episode_id:     str
    action:         ActionModel


class StepResponse(BaseModel):
    observation:    ObservationModel
    reward:         float
    terminated:     bool
    truncated:      bool
    info:           StepInfoModel


class StateResponse(BaseModel):
    state:          EpisodeStateModel


class HealthResponse(BaseModel):
    status:         str = "ok"
    version:        str = "2.0.0"
    active_episodes:int = 0
