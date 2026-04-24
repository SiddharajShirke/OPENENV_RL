"""
test_phase1.py — Gov Workflow OpenEnv v2.0
Phase 1 Validation — 40 deterministic checks across 9 sections.

Run: python test_phase1.py
All 40 checks must pass before proceeding to Phase 2.
"""

import sys
sys.path.insert(0, "/home/user/phase1")

from app.models import (
    ServiceType, InternalSubstate, DocEnrichmentType,
    EventType, ScenarioMode,
    ApplicationCase, QueueSnapshot, OfficerPool, ObservationModel,
    INTERNAL_TO_PUBLIC_STAGE,
)
from app.sector_profiles import (
    get_sector_profile, SECTOR_REGISTRY,
    get_enrichment_reason, get_enrichment_summary,
)
from app.tasks import get_task, list_benchmark_tasks
from app.event_engine import EventEngine, DayEventParams
from app.signal_computer import SignalComputer, ComputedSignals

# ─── Harness ──────────────────────────────────────────────────────────────────
PASS = "✅"; FAIL = "❌"
results = []

def check(label: str, condition: bool):
    symbol = PASS if condition else FAIL
    print(f"  {symbol} {label}")
    results.append((label, condition))

# ═══════════════════════════════════════════════════════════════════════
print("\n══ SECTION 1: Enum Completeness (6 checks) ══")
# ═══════════════════════════════════════════════════════════════════════

check("ServiceType has 7 services",
      len(list(ServiceType)) == 7)

check("InternalSubstate has BLOCKED_ENRICHMENT",
      hasattr(InternalSubstate, "BLOCKED_ENRICHMENT"))

check("InternalSubstate has BLOCKED_MISSING_DOCS",
      hasattr(InternalSubstate, "BLOCKED_MISSING_DOCS"))

check("DocEnrichmentType has 5 variants",
      len(list(DocEnrichmentType)) == 5)

check("EventType has REVENUE_DB_DELAY",
      hasattr(EventType, "REVENUE_DB_DELAY"))

check("INTERNAL_TO_PUBLIC_STAGE maps blocked_enrichment → document_verification",
      INTERNAL_TO_PUBLIC_STAGE.get("blocked_enrichment") == "document_verification")


# ═══════════════════════════════════════════════════════════════════════
print("\n══ SECTION 2: DocEnrichmentType per Service (8 checks) ══")
# ═══════════════════════════════════════════════════════════════════════

land_p   = get_sector_profile(ServiceType.LAND_REGISTRATION)
caste_p  = get_sector_profile(ServiceType.CASTE_CERTIFICATE)
income_p = get_sector_profile(ServiceType.INCOME_CERTIFICATE)
pass_p   = get_sector_profile(ServiceType.PASSPORT)
gst_p    = get_sector_profile(ServiceType.GST_REGISTRATION)

check("Land → PAST_LAND_RECORDS",
      land_p.doc_enrichment_type == DocEnrichmentType.PAST_LAND_RECORDS)

check("Caste → FAMILY_CASTE_HISTORY",
      caste_p.doc_enrichment_type == DocEnrichmentType.FAMILY_CASTE_HISTORY)

check("Income → NONE",
      income_p.doc_enrichment_type == DocEnrichmentType.NONE)

check("Passport → POLICE_VERIFICATION",
      pass_p.doc_enrichment_type == DocEnrichmentType.POLICE_VERIFICATION)

check("GST → TAX_RECORD_CROSS_CHECK",
      gst_p.doc_enrichment_type == DocEnrichmentType.TAX_RECORD_CROSS_CHECK)

check("Income enrichment probability == 0.0",
      income_p.doc_enrichment_probability == 0.0)

check("Land enrichment probability > 0",
      land_p.doc_enrichment_probability > 0.0)

check("Land enrichment delay range valid (min < max)",
      land_p.doc_enrichment_delay_days_min < land_p.doc_enrichment_delay_days_max)


# ═══════════════════════════════════════════════════════════════════════
print("\n══ SECTION 3: SectorProfile SLA Values (7 checks) ══")
# ═══════════════════════════════════════════════════════════════════════

check("Income Certificate SLA = 21 days",
      income_p.sla_days == 21)

check("Land Registration SLA = 30 days",
      land_p.sla_days == 30)

check("Caste Certificate SLA = 21 days",
      caste_p.sla_days == 21)

check("Passport SLA = 30 days",
      pass_p.sla_days == 30)

check("All 7 services in SECTOR_REGISTRY",
      len(SECTOR_REGISTRY) == 7)

check("get_enrichment_summary() returns 7 entries",
      len(get_enrichment_summary()) == 7)

check("Enrichment summary: land = past_land_records",
      get_enrichment_summary()["land_registration"]["enrichment_type"] == "past_land_records")


# ═══════════════════════════════════════════════════════════════════════
print("\n══ SECTION 4: ApplicationCase Enrichment Fields (7 checks) ══")
# ═══════════════════════════════════════════════════════════════════════

land_case = ApplicationCase(
    service_type=ServiceType.LAND_REGISTRATION,
    arrival_day=0, current_day=5, sla_deadline_day=30,
    internal_substate=InternalSubstate.BLOCKED_ENRICHMENT,
    doc_enrichment_type=DocEnrichmentType.PAST_LAND_RECORDS,
    doc_enrichment_triggered=True,
    doc_enrichment_reason="revenue_db_lookup_pending",
    enrichment_resolution_day=8,
)

check("ApplicationCase.doc_enrichment_type stored correctly",
      land_case.doc_enrichment_type == DocEnrichmentType.PAST_LAND_RECORDS)

check("ApplicationCase.doc_enrichment_triggered = True",
      land_case.doc_enrichment_triggered is True)

check("ApplicationCase.doc_enrichment_reason = correct string",
      land_case.doc_enrichment_reason == "revenue_db_lookup_pending")

check("ApplicationCase.enrichment_resolution_day = 8",
      land_case.enrichment_resolution_day == 8)

check("ApplicationCase.is_blocked = True when BLOCKED_ENRICHMENT",
      land_case.is_blocked is True)

income_case = ApplicationCase(
    service_type=ServiceType.INCOME_CERTIFICATE,
    arrival_day=0, current_day=0, sla_deadline_day=21,
)
check("Income case is_blocked = False at PRE_SCRUTINY",
      income_case.is_blocked is False)

reason = get_enrichment_reason(DocEnrichmentType.PAST_LAND_RECORDS, 0)
check("get_enrichment_reason(PAST_LAND_RECORDS) returns non-empty string",
      isinstance(reason, str) and len(reason) > 0)


# ═══════════════════════════════════════════════════════════════════════
print("\n══ SECTION 5: QueueSnapshot Counts (3 checks) ══")
# ═══════════════════════════════════════════════════════════════════════

snap = QueueSnapshot(
    service_type=ServiceType.LAND_REGISTRATION,
    total_pending=15,
    blocked_missing_docs=3,
    blocked_enrichment=6,
)

check("QueueSnapshot.blocked_enrichment = 6",
      snap.blocked_enrichment == 6)

check("QueueSnapshot.blocked_missing_docs = 3",
      snap.blocked_missing_docs == 3)

check("QueueSnapshot.total_blocked = 9 (3 + 6)",
      snap.total_blocked == 9)


# ═══════════════════════════════════════════════════════════════════════
print("\n══ SECTION 6: ObservationModel Signals (4 checks) ══")
# ═══════════════════════════════════════════════════════════════════════

obs_fields = set(ObservationModel.model_fields.keys())

check("ObservationModel has blocked_cases_enrichment",
      "blocked_cases_enrichment" in obs_fields)

check("ObservationModel has blocked_cases_missing_docs",
      "blocked_cases_missing_docs" in obs_fields)

check("ObservationModel has pending_enrichment_lookups",
      "pending_enrichment_lookups" in obs_fields)

check("ObservationModel has sla_risk_score",
      "sla_risk_score" in obs_fields)


# ═══════════════════════════════════════════════════════════════════════
print("\n══ SECTION 7: Task Configs (5 checks) ══")
# ═══════════════════════════════════════════════════════════════════════

easy   = get_task("district_backlog_easy")
medium = get_task("mixed_urgency_medium")
hard   = get_task("cross_department_hard")

check("Easy: only INCOME_CERTIFICATE enabled",
      easy.enabled_services == [ServiceType.INCOME_CERTIFICATE])

check("Medium: INCOME + LAND (2 services)",
      len(medium.enabled_services) == 2 and
      ServiceType.LAND_REGISTRATION in medium.enabled_services)

check("Hard: 3 services including CASTE_CERTIFICATE",
      len(hard.enabled_services) == 3 and
      ServiceType.CASTE_CERTIFICATE in hard.enabled_services)

check("Hard: fairness_threshold = 0.70",
      hard.fairness_threshold == 0.70)

check("Hard: REVENUE_DB_DELAY in allowed_events",
      EventType.REVENUE_DB_DELAY in hard.allowed_events)


# ═══════════════════════════════════════════════════════════════════════
print("\n══ SECTION 8: EventEngine Determinism (5 checks) ══")
# ═══════════════════════════════════════════════════════════════════════

eng_normal = EventEngine(seed=42,  scenario_mode=ScenarioMode.NORMAL)
eng_crisis = EventEngine(seed=999, scenario_mode=ScenarioMode.CRISIS)

ev_a = eng_crisis.get_events_for_day(5, hard)
ev_b = eng_crisis.get_events_for_day(5, hard)
check("EventEngine is deterministic: same seed+day = same events",
      ev_a == ev_b)

ev_easy = eng_normal.get_events_for_day(5, easy)
check("Easy task (only NO_EVENT allowed) always returns [NO_EVENT]",
      ev_easy == [EventType.NO_EVENT])

params: DayEventParams = eng_crisis.apply_events([EventType.REVENUE_DB_DELAY], hard)
check("REVENUE_DB_DELAY in CRISIS → system_dependency_boost = 2.0",
      params.system_dependency_boost == 2.0)

desc = eng_crisis.describe_events([EventType.REVENUE_DB_DELAY])
check("REVENUE_DB_DELAY description mentions 'land' AND 'caste'",
      "land" in desc and "caste" in desc)

params2: DayEventParams = eng_crisis.apply_events([EventType.OFFICER_UNAVAILABLE], hard)
check("OFFICER_UNAVAILABLE in CRISIS → officer_reduction >= 1",
      params2.officer_reduction >= 1)


# ═══════════════════════════════════════════════════════════════════════
print("\n══ SECTION 9: SignalComputer (5 checks) ══")
# ═══════════════════════════════════════════════════════════════════════

snapshots_dict = {
    "income_certificate": QueueSnapshot(
        service_type=ServiceType.INCOME_CERTIFICATE,
        total_pending=20, total_completed_today=5,
        blocked_missing_docs=3, blocked_enrichment=0,
        current_sla_risk=0.3,
    ),
    "land_registration": QueueSnapshot(
        service_type=ServiceType.LAND_REGISTRATION,
        total_pending=10, total_completed_today=2,
        blocked_missing_docs=1, blocked_enrichment=6,
        current_sla_risk=0.5,
    ),
}
pool = OfficerPool(
    total_officers=10, available_officers=10,
    allocated={"income_certificate": 6, "land_registration": 4},
)
sc   = SignalComputer()
sigs: ComputedSignals = sc.compute(
    queue_snapshots  = snapshots_dict,
    officer_pool     = pool,
    todays_arrivals  = 12,
    digital_arrivals = 9,
    capacity_per_day = 8.0,
)

check("blocked_cases_enrichment = 6",
      sigs.blocked_cases_enrichment == 6)

check("blocked_cases_missing_docs = 4 (3 income + 1 land)",
      sigs.blocked_cases_missing_docs == 4)

check("resource_utilization = 1.0 (10/10 allocated)",
      sigs.resource_utilization == 1.0)

check("digital_intake_ratio ≈ 0.75 (9/12)",
      abs(sigs.digital_intake_ratio - 0.75) < 0.01)

check("sla_risk_score in [0.0, 1.0]",
      0.0 <= sigs.sla_risk_score <= 1.0)


# ═══════════════════════════════════════════════════════════════════════
print("\n══ FINAL RESULTS ══")
# ═══════════════════════════════════════════════════════════════════════

passed = sum(1 for _, ok in results if ok)
failed = sum(1 for _, ok in results if not ok)
total  = len(results)

print(f"\n  Checks run  : {total}")
print(f"  {PASS} Passed    : {passed}")

if failed:
    print(f"  {FAIL} Failed    : {failed}")
    print("\n  Failed checks:")
    for label, ok in results:
        if not ok:
            print(f"    → {label}")
    sys.exit(1)
else:
    print(f"\n  🏛️  ALL {total} CHECKS PASSED — Phase 1 is solid.")
    print("  Proceed to Phase 2: env.py + simulator.py + state_machine.py\n")
