from __future__ import annotations
from app.models import ServiceType, TaskConfig


TASKS: dict[str, TaskConfig] = {
    "district_backlog_easy": TaskConfig(
        task_id="district_backlog_easy",
        seed=11,
        max_days=20,
        arrival_rate=2.0,
        services=[
            ServiceType.BIRTH_CERTIFICATE,
            ServiceType.INCOME_CERTIFICATE,
            ServiceType.DRIVING_LICENSE,
        ],
        missing_docs_probability=0.18,
        field_verification_probability=0.12,
        escalation_budget=3,
        officer_pool={
            ServiceType.BIRTH_CERTIFICATE: 2,
            ServiceType.INCOME_CERTIFICATE: 2,
            ServiceType.DRIVING_LICENSE: 1,
        },
        reserve_officers=1,
        initial_cases_by_service={
            ServiceType.BIRTH_CERTIFICATE: 4,
            ServiceType.INCOME_CERTIFICATE: 4,
            ServiceType.DRIVING_LICENSE: 3,
        },
        sla_days={
            ServiceType.BIRTH_CERTIFICATE: 6,
            ServiceType.INCOME_CERTIFICATE: 7,
            ServiceType.DRIVING_LICENSE: 10,
        },
    ),
    "mixed_urgency_medium": TaskConfig(
        task_id="mixed_urgency_medium",
        seed=22,
        max_days=28,
        arrival_rate=3.0,
        services=[
            ServiceType.PASSPORT,
            ServiceType.DRIVING_LICENSE,
            ServiceType.GST_REGISTRATION,
            ServiceType.INCOME_CERTIFICATE,
        ],
        missing_docs_probability=0.24,
        field_verification_probability=0.28,
        escalation_budget=4,
        officer_pool={
            ServiceType.PASSPORT: 2,
            ServiceType.DRIVING_LICENSE: 2,
            ServiceType.GST_REGISTRATION: 2,
            ServiceType.INCOME_CERTIFICATE: 1,
        },
        reserve_officers=1,
        initial_cases_by_service={
            ServiceType.PASSPORT: 5,
            ServiceType.DRIVING_LICENSE: 5,
            ServiceType.GST_REGISTRATION: 4,
            ServiceType.INCOME_CERTIFICATE: 4,
        },
        sla_days={
            ServiceType.PASSPORT: 12,
            ServiceType.DRIVING_LICENSE: 10,
            ServiceType.GST_REGISTRATION: 9,
            ServiceType.INCOME_CERTIFICATE: 7,
        },
        fairness_threshold=0.35,
    ),
    "cross_department_hard": TaskConfig(
        task_id="cross_department_hard",
        seed=33,
        max_days=35,
        arrival_rate=4.0,
        services=[
            ServiceType.PASSPORT,
            ServiceType.GST_REGISTRATION,
            ServiceType.CASTE_CERTIFICATE,
            ServiceType.LAND_REGISTRATION,
            ServiceType.BIRTH_CERTIFICATE,
        ],
        missing_docs_probability=0.30,
        field_verification_probability=0.42,
        escalation_budget=5,
        officer_pool={
            ServiceType.PASSPORT: 2,
            ServiceType.GST_REGISTRATION: 2,
            ServiceType.CASTE_CERTIFICATE: 1,
            ServiceType.LAND_REGISTRATION: 1,
            ServiceType.BIRTH_CERTIFICATE: 2,
        },
        reserve_officers=2,
        initial_cases_by_service={
            ServiceType.PASSPORT: 6,
            ServiceType.GST_REGISTRATION: 5,
            ServiceType.CASTE_CERTIFICATE: 5,
            ServiceType.LAND_REGISTRATION: 4,
            ServiceType.BIRTH_CERTIFICATE: 5,
        },
        sla_days={
            ServiceType.PASSPORT: 12,
            ServiceType.GST_REGISTRATION: 9,
            ServiceType.CASTE_CERTIFICATE: 8,
            ServiceType.LAND_REGISTRATION: 15,
            ServiceType.BIRTH_CERTIFICATE: 6,
        },
        fairness_threshold=0.25,
    ),
}


def get_task(task_id: str) -> TaskConfig:
    if task_id not in TASKS:
        raise KeyError(f"Unknown task_id: {task_id}")
    return TASKS[task_id].model_copy(deep=True)


def list_tasks() -> list[str]:
    return sorted(TASKS.keys())