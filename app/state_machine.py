from __future__ import annotations
from app.models import ServiceCase, StageType
from app.utils import next_stage

BLOCKED_STAGES = {StageType.SUBMISSION, StageType.DOCUMENT_VERIFICATION}


def can_advance(case: ServiceCase) -> bool:
    if case.completed:
        return False
    if case.has_missing_documents and case.stage in BLOCKED_STAGES:
        return False
    return True


def advance_case(case: ServiceCase) -> tuple[bool, bool]:
    if not can_advance(case):
        return False, False
    if case.stage == StageType.ISSUANCE:
        case.completed = True
        return True, True
    case.stage = next_stage(case.stage, case.field_verification_required)
    case.days_in_stage = 0
    return True, False