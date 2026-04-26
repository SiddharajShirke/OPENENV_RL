"""
tasks.py — Gov Workflow OpenEnv v2.0
Three deterministic benchmark tasks: easy, medium, hard.
"""
from app.models import (
    TaskConfig, ServiceType, ScenarioMode, EventType, OfficerPool
)

TASK_EASY = TaskConfig(
    task_id="district_backlog_easy",
    display_name="District Backlog Clearance — Revenue Office",
    difficulty="easy",
    scenario_mode=ScenarioMode.NORMAL,
    seed=42,
    max_days=30,
    enabled_services=[ServiceType.INCOME_CERTIFICATE],
    arrival_rate_per_day={ServiceType.INCOME_CERTIFICATE: 12.0},
    digital_intake_ratio=0.65,
    initial_officer_pool=OfficerPool(
        total_officers=8, available_officers=8,
        allocated={ServiceType.INCOME_CERTIFICATE: 8},
    ),
    missing_docs_probability_override={ServiceType.INCOME_CERTIFICATE: 0.20},
    field_verification_probability_override={ServiceType.INCOME_CERTIFICATE: 0.15},
    escalation_budget=5,
    fairness_threshold=None,
    event_probability=0.05,
    allowed_events=[EventType.NO_EVENT],
)

TASK_MEDIUM = TaskConfig(
    task_id="mixed_urgency_medium",
    display_name="Mixed Urgency Backlog — Taluka Office",
    difficulty="medium",
    scenario_mode=ScenarioMode.NORMAL,
    seed=123,
    max_days=45,
    enabled_services=[
        ServiceType.INCOME_CERTIFICATE,
        ServiceType.LAND_REGISTRATION,
        ServiceType.PASSPORT,
        ServiceType.DRIVING_LICENSE,
        ServiceType.AADHAAR_CARD,
    ],
    arrival_rate_per_day={
        ServiceType.INCOME_CERTIFICATE: 8.0,
        ServiceType.LAND_REGISTRATION:  4.0,
        ServiceType.PASSPORT:           4.0,
        ServiceType.DRIVING_LICENSE:    5.0,
        ServiceType.AADHAAR_CARD:       6.0,
    },
    digital_intake_ratio=0.72,
    initial_officer_pool=OfficerPool(
        total_officers=14, available_officers=14,
        allocated={
            ServiceType.INCOME_CERTIFICATE: 4,
            ServiceType.LAND_REGISTRATION:  2,
            ServiceType.PASSPORT:           2,
            ServiceType.DRIVING_LICENSE:    3,
            ServiceType.AADHAAR_CARD:       3,
        },
    ),
    missing_docs_probability_override=None,
    field_verification_probability_override=None,
    escalation_budget=8,
    fairness_threshold=None,
    event_probability=0.15,
    allowed_events=[EventType.DOCUMENT_REJECTION_SPIKE],
)

TASK_HARD = TaskConfig(
    task_id="cross_department_hard",
    display_name="Cross-Department Crisis — District Collectorate",
    difficulty="hard",
    scenario_mode=ScenarioMode.CRISIS,
    seed=999,
    max_days=60,
    enabled_services=[
        ServiceType.INCOME_CERTIFICATE,
        ServiceType.LAND_REGISTRATION,
        ServiceType.PASSPORT,
        ServiceType.DRIVING_LICENSE,
        ServiceType.AADHAAR_CARD,
    ],
    arrival_rate_per_day={
        ServiceType.INCOME_CERTIFICATE: 11.0,
        ServiceType.LAND_REGISTRATION:  6.0,
        ServiceType.PASSPORT:           6.0,
        ServiceType.DRIVING_LICENSE:    7.0,
        ServiceType.AADHAAR_CARD:       8.0,
    },
    digital_intake_ratio=0.80,
    initial_officer_pool=OfficerPool(
        total_officers=18, available_officers=18,
        allocated={
            ServiceType.INCOME_CERTIFICATE: 5,
            ServiceType.LAND_REGISTRATION:  3,
            ServiceType.PASSPORT:           3,
            ServiceType.DRIVING_LICENSE:    3,
            ServiceType.AADHAAR_CARD:       4,
        },
    ),
    missing_docs_probability_override=None,
    field_verification_probability_override=None,
    escalation_budget=10,
    fairness_threshold=0.70,
    event_probability=0.30,
    allowed_events=[
        EventType.SURGE_APPLICATIONS,
        EventType.OFFICER_UNAVAILABLE,
        EventType.DOCUMENT_REJECTION_SPIKE,
        EventType.REVENUE_DB_DELAY,
        EventType.SLA_ESCALATION_ORDER,
    ],
)

def make_extreme_variant(base_task: TaskConfig) -> TaskConfig:
    variant = base_task.model_copy(deep=True)
    variant.task_id = base_task.task_id + "_extreme"
    variant.display_name = base_task.display_name + " [EXTREME]"
    variant.scenario_mode = ScenarioMode.EXTREME_OVERLOAD
    variant.event_probability = min(1.0, base_task.event_probability * 3.0)
    variant.allowed_events = [e for e in EventType if e != EventType.NO_EVENT]
    return variant

TASK_REGISTRY: dict = {
    "district_backlog_easy":         TASK_EASY,
    "mixed_urgency_medium":          TASK_MEDIUM,
    "cross_department_hard":         TASK_HARD,
    "district_backlog_easy_extreme": make_extreme_variant(TASK_EASY),
}

def get_task(task_id: str) -> TaskConfig:
    if task_id not in TASK_REGISTRY:
        raise ValueError(f"Unknown task_id '{task_id}'. Available: {list(TASK_REGISTRY)}")
    return TASK_REGISTRY[task_id]

def list_tasks() -> list:
    return list(TASK_REGISTRY.keys())

def list_benchmark_tasks() -> list:
    return ["district_backlog_easy", "mixed_urgency_medium", "cross_department_hard"]

TASKS = TASK_REGISTRY
