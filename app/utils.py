from __future__ import annotations
from app.models import PriorityMode, ServiceCase, ServiceType, StageType


def next_stage(current: StageType, field_verification_required: bool) -> StageType:
    if current == StageType.SUBMISSION:
        return StageType.DOCUMENT_VERIFICATION
    if current == StageType.DOCUMENT_VERIFICATION:
        return StageType.FIELD_VERIFICATION if field_verification_required else StageType.APPROVAL
    if current == StageType.FIELD_VERIFICATION:
        return StageType.APPROVAL
    return StageType.ISSUANCE


def days_to_sla(case: ServiceCase, current_day: int) -> int:
    return case.due_day - current_day


def priority_key(case: ServiceCase, mode: PriorityMode, current_day: int) -> tuple:
    remaining = days_to_sla(case, current_day)
    if mode == PriorityMode.URGENT_FIRST:
        return (-int(case.escalated), -case.urgency, remaining, -case.total_days, case.case_id)
    if mode == PriorityMode.OLDEST_FIRST:
        return (-int(case.escalated), -case.total_days, remaining, -case.urgency, case.case_id)
    if mode == PriorityMode.BACKLOG_CLEARANCE:
        return (-int(case.escalated), case.stage.value, remaining, -case.total_days, case.case_id)
    # BALANCED default
    return (-int(case.escalated), remaining, -case.urgency, -case.total_days, case.case_id)


def completion_fairness_gap(
    arrived: dict[ServiceType, int],
    completed: dict[ServiceType, int],
) -> float:
    rates: list[float] = []
    for service, total in arrived.items():
        if total > 0:
            rates.append(completed.get(service, 0) / total)
    if len(rates) < 2:
        return 0.0
    return round(max(rates) - min(rates), 4)


def bounded_score(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 4)