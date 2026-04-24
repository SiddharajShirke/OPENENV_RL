"""
state_machine.py — Gov Workflow OpenEnv
Deterministic workflow transition engine aligned with Phase 1 schemas.
"""

from __future__ import annotations

from app.models import ApplicationCase, InternalSubstate, StageType


INTERNAL_TO_PUBLIC_STAGE: dict[InternalSubstate, StageType] = {
    InternalSubstate.PRE_SCRUTINY: StageType.SUBMISSION,
    InternalSubstate.DOC_VALIDATION: StageType.DOCUMENT_VERIFICATION,
    InternalSubstate.SERVICE_SPECIFIC_VALIDATION: StageType.DOCUMENT_VERIFICATION,
    InternalSubstate.FIELD_VERIFICATION_PENDING: StageType.FIELD_VERIFICATION,
    InternalSubstate.DECISION_PENDING: StageType.APPROVAL,
    InternalSubstate.ISSUANCE_READY: StageType.ISSUANCE,
    InternalSubstate.BLOCKED_MISSING_DOCS: StageType.DOCUMENT_VERIFICATION,
    InternalSubstate.COMPLETED: StageType.ISSUANCE,
    InternalSubstate.REJECTED: StageType.APPROVAL,
}


def build_public_stage(substate: InternalSubstate) -> StageType:
    return INTERNAL_TO_PUBLIC_STAGE.get(substate, StageType.SUBMISSION)


def transition_case(case: ApplicationCase, new_substate: InternalSubstate) -> None:
    case.internal_substate = new_substate
    case.public_stage = build_public_stage(new_substate)
    case.days_in_current_stage = 0


def can_advance(case: ApplicationCase) -> bool:
    if case.completed or case.rejected:
        return False
    if case.internal_substate == InternalSubstate.BLOCKED_MISSING_DOCS:
        return False
    return True


def advance_case(case: ApplicationCase, rng: object = None) -> tuple[bool, bool]:
    """
    Returns (progressed, completed).
    """
    if not can_advance(case):
        return False, False

    early_stages = {
        InternalSubstate.PRE_SCRUTINY,
        InternalSubstate.DOC_VALIDATION,
    }

    if case.has_missing_docs and case.internal_substate in early_stages:
        transition_case(case, InternalSubstate.BLOCKED_MISSING_DOCS)
        return True, False

    current = case.internal_substate

    if current == InternalSubstate.PRE_SCRUTINY:
        transition_case(case, InternalSubstate.DOC_VALIDATION)
        return True, False

    if current == InternalSubstate.DOC_VALIDATION:
        if case.field_verification_required:
            transition_case(case, InternalSubstate.FIELD_VERIFICATION_PENDING)
        else:
            transition_case(case, InternalSubstate.DECISION_PENDING)
        return True, False

    if current == InternalSubstate.SERVICE_SPECIFIC_VALIDATION:
        if case.field_verification_required:
            transition_case(case, InternalSubstate.FIELD_VERIFICATION_PENDING)
        else:
            transition_case(case, InternalSubstate.DECISION_PENDING)
        return True, False

    if current == InternalSubstate.FIELD_VERIFICATION_PENDING:
        return False, False

    if current == InternalSubstate.DECISION_PENDING:
        transition_case(case, InternalSubstate.ISSUANCE_READY)
        return True, False

    if current == InternalSubstate.ISSUANCE_READY:
        transition_case(case, InternalSubstate.COMPLETED)
        case.completed = True
        return True, True

    return False, False


def unblock_missing_docs(case: ApplicationCase) -> bool:
    if case.internal_substate != InternalSubstate.BLOCKED_MISSING_DOCS:
        return False
    case.has_missing_docs = False
    case.doc_resolution_day = None
    transition_case(case, InternalSubstate.DOC_VALIDATION)
    return True


def complete_field_verification(case: ApplicationCase) -> bool:
    if case.internal_substate != InternalSubstate.FIELD_VERIFICATION_PENDING:
        return False
    case.field_verification_completion_day = None
    transition_case(case, InternalSubstate.DECISION_PENDING)
    return True