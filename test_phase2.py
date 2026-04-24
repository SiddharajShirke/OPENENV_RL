"""
test_phase2.py — Gov Workflow OpenEnv v2.0
End-to-end validation test suite for Phase 2.

Run with:
    python test_phase2.py
"""
from __future__ import annotations
import sys
import traceback
import random
from typing import Callable

# ─────────────────────────────────────────────────────────────────────────────
# Test runner — always prints every PASS/FAIL
# ─────────────────────────────────────────────────────────────────────────────

PASS = "PASS"
FAIL = "FAIL"
_results: list = []


def test(section: str, name: str, fn: Callable) -> bool:
    try:
        fn()
        _results.append((section, name, PASS))
        print(f"  [PASS]  {name}")
        return True
    except AssertionError as exc:
        _results.append((section, name, f"{FAIL}: {exc}"))
        print(f"  [FAIL]  {name}")
        print(f"          AssertionError: {exc}")
        return False
    except Exception as exc:
        _results.append((section, name, f"{FAIL}: {type(exc).__name__}: {exc}"))
        print(f"  [FAIL]  {name}")
        print(f"          {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return False


def section(title: str):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def summary():
    total  = len(_results)
    passed = sum(1 for _, _, s in _results if s == PASS)
    failed = total - passed
    print(f"\n{'=' * 60}")
    if failed == 0:
        print(f"  RESULTS: {passed}/{total} passed  <<< ALL CLEAR >>>")
    else:
        print(f"  RESULTS: {passed}/{total} passed  <<< {failed} FAILED >>>")
        print(f"\n  Failed tests:")
        for sec, name, status in _results:
            if status != PASS:
                print(f"    [{sec}] {name}")
                print(f"           {status}")
    print(f"{'=' * 60}")
    sys.exit(0 if failed == 0 else 1)


# =============================================================================
# T1 — MODELS & ENUMS
# =============================================================================
section("T1 — Models & Enums")


def t1_service_type_values():
    from app.models import ServiceType
    expected = {
        "passport", "driving_license", "gst_registration",
        "income_certificate", "caste_certificate",
        "birth_certificate", "land_registration",
    }
    actual = {s.value for s in ServiceType}
    missing = expected - actual
    assert not missing, f"Missing ServiceType values: {missing}"


def t1_stage_type_values():
    from app.models import StageType
    expected = {
        "submission", "document_verification",
        "field_verification", "approval", "issuance",
    }
    actual = {s.value for s in StageType}
    assert actual == expected, f"StageType mismatch: {actual ^ expected}"


def t1_internal_substate_has_blocked_enrichment():
    from app.models import InternalSubstate
    values = {s.value for s in InternalSubstate}
    assert "blocked_enrichment" in values, (
        "InternalSubstate must have blocked_enrichment (Phase 2)"
    )


def t1_doc_enrichment_type_exists():
    from app.models import DocEnrichmentType
    assert DocEnrichmentType.PAST_LAND_RECORDS is not None
    assert DocEnrichmentType.POLICE_VERIFICATION is not None
    assert DocEnrichmentType.NONE is not None
    assert DocEnrichmentType.FAMILY_CASTE_HISTORY is not None


def t1_internal_to_public_stage_mapping():
    from app.models import INTERNAL_TO_PUBLIC_STAGE
    assert "blocked_enrichment" in INTERNAL_TO_PUBLIC_STAGE, (
        "INTERNAL_TO_PUBLIC_STAGE must map blocked_enrichment"
    )
    assert INTERNAL_TO_PUBLIC_STAGE["blocked_enrichment"] == "document_verification"
    assert INTERNAL_TO_PUBLIC_STAGE["completed"] == "issuance"
    assert INTERNAL_TO_PUBLIC_STAGE["pre_scrutiny"] == "submission"


def t1_application_case_enrichment_fields():
    from app.models import ApplicationCase, ServiceType, DocEnrichmentType
    case = ApplicationCase(
        service_type=ServiceType.LAND_REGISTRATION,
        arrival_day=0, current_day=0, sla_deadline_day=30,
        doc_enrichment_type=DocEnrichmentType.PAST_LAND_RECORDS,
        doc_enrichment_triggered=True,
        enrichment_resolution_day=5,
    )
    assert case.doc_enrichment_type == DocEnrichmentType.PAST_LAND_RECORDS
    assert case.doc_enrichment_triggered is True
    assert case.enrichment_resolution_day == 5


def t1_observation_model_enrichment_fields():
    from app.models import ObservationModel, OfficerPool, ScenarioMode
    obs = ObservationModel(
        task_id="test", episode_id="ep-001",
        day=0, max_days=30,
        scenario_mode=ScenarioMode.NORMAL,
        officer_pool=OfficerPool(total_officers=5, available_officers=5),
        blocked_cases_enrichment=3,
        pending_enrichment_lookups=2,
    )
    assert obs.blocked_cases_enrichment == 3
    assert obs.pending_enrichment_lookups == 2


def t1_queue_snapshot_blocked_enrichment_field():
    from app.models import QueueSnapshot, ServiceType
    snap = QueueSnapshot(
        service_type=ServiceType.LAND_REGISTRATION,
        blocked_enrichment=4,
    )
    assert snap.blocked_enrichment == 4


def t1_sector_profile_enrichment_fields():
    from app.models import SectorProfile, ServiceType, UrgencyProfile, DocEnrichmentType
    profile = SectorProfile(
        service_type=ServiceType.LAND_REGISTRATION,
        sector_name="Test Land",
        missing_docs_probability=0.3,
        doc_defect_rate_digital=0.2,
        doc_defect_rate_paper=0.5,
        field_verification_probability=0.6,
        manual_scrutiny_intensity=0.7,
        decision_backlog_sensitivity=0.8,
        system_dependency_risk=0.5,
        sla_days=30,
        urgency_profile=UrgencyProfile.LOW,
        base_processing_rate=4.0,
        field_verification_days=5,
        doc_enrichment_type=DocEnrichmentType.PAST_LAND_RECORDS,
        doc_enrichment_probability=0.70,
        doc_enrichment_delay_days_min=2,
        doc_enrichment_delay_days_max=5,
    )
    assert profile.doc_enrichment_type == DocEnrichmentType.PAST_LAND_RECORDS
    assert profile.doc_enrichment_probability == 0.70


def t1_sla_risk_property():
    from app.models import ApplicationCase, ServiceType
    case = ApplicationCase(
        service_type=ServiceType.INCOME_CERTIFICATE,
        arrival_day=0, current_day=10, sla_deadline_day=21,
    )
    risk = case.sla_risk
    assert 0.0 <= risk <= 1.0, f"sla_risk out of range: {risk}"
    expected = 10 / 21
    assert abs(risk - expected) < 0.01, f"Expected ~{expected:.3f}, got {risk:.3f}"


for fn in [
    t1_service_type_values, t1_stage_type_values,
    t1_internal_substate_has_blocked_enrichment,
    t1_doc_enrichment_type_exists,
    t1_internal_to_public_stage_mapping,
    t1_application_case_enrichment_fields,
    t1_observation_model_enrichment_fields,
    t1_queue_snapshot_blocked_enrichment_field,
    t1_sector_profile_enrichment_fields,
    t1_sla_risk_property,
]:
    test("T1", fn.__name__, fn)


# =============================================================================
# T2 — SECTOR PROFILES
# =============================================================================
section("T2 — Sector Profiles")


def t2_all_services_have_profiles():
    from app.models import ServiceType
    from app.sector_profiles import get_sector_profile
    for svc in ServiceType:
        profile = get_sector_profile(svc)
        assert profile is not None, f"No profile for {svc.value}"


def t2_income_cert_no_enrichment():
    from app.models import ServiceType, DocEnrichmentType
    from app.sector_profiles import get_sector_profile
    p = get_sector_profile(ServiceType.INCOME_CERTIFICATE)
    assert p.doc_enrichment_type == DocEnrichmentType.NONE
    assert p.doc_enrichment_probability == 0.0
    assert p.sla_days == 21


def t2_land_reg_has_enrichment():
    from app.models import ServiceType, DocEnrichmentType
    from app.sector_profiles import get_sector_profile
    p = get_sector_profile(ServiceType.LAND_REGISTRATION)
    assert p.doc_enrichment_type == DocEnrichmentType.PAST_LAND_RECORDS
    assert p.doc_enrichment_probability > 0.5
    assert p.doc_enrichment_delay_days_min >= 2
    assert p.doc_enrichment_delay_days_max >= p.doc_enrichment_delay_days_min


def t2_passport_police_verification():
    from app.models import ServiceType, DocEnrichmentType
    from app.sector_profiles import get_sector_profile
    p = get_sector_profile(ServiceType.PASSPORT)
    assert p.doc_enrichment_type == DocEnrichmentType.POLICE_VERIFICATION
    assert p.field_verification_days >= 14, (
        f"Passport police check should take >=14 days, got {p.field_verification_days}"
    )


def t2_birth_cert_fast_sla():
    from app.models import ServiceType
    from app.sector_profiles import get_sector_profile
    p = get_sector_profile(ServiceType.BIRTH_CERTIFICATE)
    assert p.sla_days <= 7, f"Birth cert SLA should be <=7 days, got {p.sla_days}"


def t2_probabilities_in_range():
    from app.models import ServiceType
    from app.sector_profiles import get_sector_profile
    for svc in ServiceType:
        p = get_sector_profile(svc)
        assert 0.0 <= p.missing_docs_probability <= 1.0, f"{svc.value}: missing_docs out of range"
        assert 0.0 <= p.field_verification_probability <= 1.0, f"{svc.value}: fv_prob out of range"
        assert 0.0 <= p.doc_enrichment_probability <= 1.0, f"{svc.value}: enrichment_prob out of range"


for fn in [
    t2_all_services_have_profiles,
    t2_income_cert_no_enrichment,
    t2_land_reg_has_enrichment,
    t2_passport_police_verification,
    t2_birth_cert_fast_sla,
    t2_probabilities_in_range,
]:
    test("T2", fn.__name__, fn)


# =============================================================================
# T3 — TASKS REGISTRY
# =============================================================================
section("T3 — Tasks Registry")


def t3_all_benchmark_tasks_loadable():
    from app.tasks import list_benchmark_tasks, get_task
    for tid in list_benchmark_tasks():
        t = get_task(tid)
        assert t.task_id == tid, f"task_id mismatch: {t.task_id} != {tid}"


def t3_task_seeds_deterministic():
    from app.tasks import get_task
    easy1 = get_task("district_backlog_easy")
    easy2 =