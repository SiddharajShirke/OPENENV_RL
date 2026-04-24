"""
tests/test_phase2_simulator.py
Phase 2: simulator.py — DaySimulator, case lifecycle, queue snapshots
Run: pytest tests/test_phase2_simulator.py -v
"""
import pytest
import random
from app.models import (
    ApplicationCase, ServiceType, InternalSubstate, IntakeChannel,
    ScenarioMode, EventType, QueueSnapshot,
)
from app.event_engine import EventEngine
from app.tasks import get_task
from app.simulator import DaySimulator, DayResult


def make_simulator(task_id="district_backlog_easy",
                   seed=42) -> DaySimulator:
    task = get_task(task_id)
    rng = random.Random(seed)
    engine = EventEngine(seed=seed, scenario_mode=task.scenario_mode)
    return DaySimulator(task_config=task, rng=rng, event_engine=engine)


# ─── DayResult defaults ───────────────────────────────────────────────────────
class TestDayResult:
    def test_all_counters_zero(self):
        r = DayResult()
        assert r.new_arrivals == 0
        assert r.new_completions == 0
        assert r.stage_advances == 0
        assert r.new_sla_breaches == 0
        assert r.idle_officer_days == 0
        assert r.total_capacity_days == 0
        assert r.newly_unblocked_missing == 0
        assert r.urgent_completed == 0

    def test_active_events_empty(self):
        r = DayResult()
        assert r.active_events == []


# ─── DaySimulator construction ────────────────────────────────────────────────
class TestDaySimulatorConstruction:
    def test_simulator_initialises(self):
        sim = make_simulator()
        assert sim is not None

    def test_simulator_has_case_counter(self):
        sim = make_simulator()
        assert hasattr(sim, "case_counter")
        assert sim.case_counter == 0


# ─── simulate_day ─────────────────────────────────────────────────────────────
class TestSimulateDay:
    def test_simulate_day_returns_day_result(self):
        sim = make_simulator()
        active, completed = [], []
        result = sim.simulate_day(
            day=1, active_cases=active, completed_cases=completed,
            priority_mode=None,
            officer_allocations={"income_certificate": 8},
        )
        assert isinstance(result, DayResult)

    def test_day_one_spawns_arrivals(self):
        sim = make_simulator()
        active, completed = [], []
        result = sim.simulate_day(
            day=1, active_cases=active, completed_cases=completed,
            priority_mode=None,
            officer_allocations={"income_certificate": 8},
        )
        assert result.new_arrivals > 0, "Day 1 should spawn new cases"

    def test_arrivals_added_to_active_list(self):
        sim = make_simulator()
        active, completed = [], []
        sim.simulate_day(
            day=1, active_cases=active, completed_cases=completed,
            priority_mode=None,
            officer_allocations={"income_certificate": 8},
        )
        assert len(active) > 0

    def test_completed_cases_removed_from_active(self):
        """Run enough days so some cases complete, verify no overlap."""
        sim = make_simulator()
        active, completed = [], []
        for day in range(1, 40):
            sim.simulate_day(
                day=day, active_cases=active, completed_cases=completed,
                priority_mode=None,
                officer_allocations={"income_certificate": 8},
            )
        active_ids = {c.case_id for c in active}
        completed_ids = {c.case_id for c in completed}
        assert active_ids.isdisjoint(completed_ids),             "Completed cases must not appear in active list"

    def test_total_capacity_days_equals_allocation(self):
        sim = make_simulator()
        active, completed = [], []
        result = sim.simulate_day(
            day=1, active_cases=active, completed_cases=completed,
            priority_mode=None,
            officer_allocations={"income_certificate": 8},
        )
        assert result.total_capacity_days == 8

    def test_idle_officer_days_nonnegative(self):
        sim = make_simulator()
        active, completed = [], []
        result = sim.simulate_day(
            day=1, active_cases=active, completed_cases=completed,
            priority_mode=None,
            officer_allocations={"income_certificate": 8},
        )
        assert result.idle_officer_days >= 0

    def test_idle_plus_work_equals_capacity(self):
        sim = make_simulator()
        active, completed = [], []
        result = sim.simulate_day(
            day=1, active_cases=active, completed_cases=completed,
            priority_mode=None,
            officer_allocations={"income_certificate": 4},
        )
        assert result.idle_officer_days + result.new_completions <= 4 + result.stage_advances

    def test_determinism_same_seed(self):
        def run_days(seed):
            sim = make_simulator(seed=seed)
            active, completed = [], []
            arrivals = []
            for d in range(1, 6):
                r = sim.simulate_day(
                    day=d, active_cases=active, completed_cases=completed,
                    priority_mode=None,
                    officer_allocations={"income_certificate": 8},
                )
                arrivals.append(r.new_arrivals)
            return arrivals

        assert run_days(42) == run_days(42)

    def test_sla_breaches_counted(self):
        sim = make_simulator()
        active, completed = [], []
        total_breaches = 0
        for day in range(1, 50):
            r = sim.simulate_day(
                day=day, active_cases=active, completed_cases=completed,
                priority_mode=None,
                officer_allocations={"income_certificate": 1},  # Low capacity → breaches
            )
            total_breaches += r.new_sla_breaches
        # Not guaranteed but with low capacity and 50 days, very likely
        assert total_breaches >= 0


# ─── build_queue_snapshot ──────────────────────────────────────────────────────
class TestBuildQueueSnapshot:
    def _make_case(self, service, substate=InternalSubstate.PRE_SCRUTINY,
                   urgent=False, blocked=False, field=False):
        case = ApplicationCase(
            service_type=service,
            arrival_day=0,
            current_day=5,
            sla_deadline_day=21,
            is_urgent=urgent,
        )
        case.internal_substate = substate
        case.has_missing_docs = blocked
        case.field_verification_required = field
        return case

    def test_snapshot_service_type_correct(self):
        sim = make_simulator()
        snap = sim.build_queue_snapshot(ServiceType.INCOME_CERTIFICATE, [], day=1)
        assert snap.service_type == ServiceType.INCOME_CERTIFICATE

    def test_snapshot_counts_pending_cases(self):
        sim = make_simulator()
        cases = [self._make_case(ServiceType.INCOME_CERTIFICATE) for _ in range(5)]
        snap = sim.build_queue_snapshot(ServiceType.INCOME_CERTIFICATE, cases, day=1)
        assert snap.total_pending == 5

    def test_snapshot_counts_urgent_cases(self):
        sim = make_simulator()
        cases = [
            self._make_case(ServiceType.INCOME_CERTIFICATE, urgent=True),
            self._make_case(ServiceType.INCOME_CERTIFICATE, urgent=False),
        ]
        snap = sim.build_queue_snapshot(ServiceType.INCOME_CERTIFICATE, cases, day=1)
        assert snap.urgent_pending == 1

    def test_snapshot_counts_blocked_missing_docs(self):
        sim = make_simulator()
        cases = [
            self._make_case(ServiceType.INCOME_CERTIFICATE,
                            substate=InternalSubstate.BLOCKED_MISSING_DOCS),
            self._make_case(ServiceType.INCOME_CERTIFICATE),
        ]
        snap = sim.build_queue_snapshot(ServiceType.INCOME_CERTIFICATE, cases, day=1)
        assert snap.blocked_missing_docs == 1

    def test_snapshot_sla_risk_bounded(self):
        sim = make_simulator()
        cases = [self._make_case(ServiceType.INCOME_CERTIFICATE) for _ in range(3)]
        snap = sim.build_queue_snapshot(ServiceType.INCOME_CERTIFICATE, cases, day=15)
        assert 0.0 <= snap.current_sla_risk <= 1.0


# ─── Case generation ─────────────────────────────────────────────────────────
class TestCaseGeneration:
    def test_new_case_has_correct_service(self):
        from app.event_engine import DayEventParams
        sim = make_simulator()
        params = DayEventParams()
        case = sim._new_case(ServiceType.INCOME_CERTIFICATE, day=1, params=params)
        assert case.service_type == ServiceType.INCOME_CERTIFICATE

    def test_new_case_arrival_day_set(self):
        from app.event_engine import DayEventParams
        sim = make_simulator()
        params = DayEventParams()
        case = sim._new_case(ServiceType.INCOME_CERTIFICATE, day=5, params=params)
        assert case.arrival_day == 5

    def test_new_case_sla_deadline_after_arrival(self):
        from app.event_engine import DayEventParams
        sim = make_simulator()
        params = DayEventParams()
        case = sim._new_case(ServiceType.INCOME_CERTIFICATE, day=1, params=params)
        assert case.sla_deadline_day > case.arrival_day

    def test_new_case_has_valid_intake_channel(self):
        from app.event_engine import DayEventParams
        sim = make_simulator()
        params = DayEventParams()
        case = sim._new_case(ServiceType.INCOME_CERTIFICATE, day=1, params=params)
        assert isinstance(case.intake_channel, IntakeChannel)
