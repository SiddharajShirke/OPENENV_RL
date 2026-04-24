"""
tests/test_phase1_event_engine.py
Phase 1 validation: event_engine.py — determinism, scenario scaling, event effects
Run: pytest tests/test_phase1_event_engine.py -v
"""
import pytest
from app.models import EventType, ScenarioMode
from app.event_engine import EventEngine, DayEventParams, SCENARIO_MULTIPLIER, BASE_PROBS
from app.tasks import get_task


# ─── DayEventParams defaults ─────────────────────────────────────────────────
class TestDayEventParams:
    def test_default_arrival_multiplier_one(self):
        p = DayEventParams()
        assert p.arrival_multiplier == 1.0

    def test_default_officer_reduction_zero(self):
        p = DayEventParams()
        assert p.officer_reduction == 0

    def test_default_no_active_events(self):
        p = DayEventParams()
        assert p.active_events == []

    def test_has_events_false_by_default(self):
        p = DayEventParams()
        assert p.has_events() is False

    def test_has_events_true_when_populated(self):
        p = DayEventParams()
        p.active_events.append(EventType.SURGE_APPLICATIONS)
        assert p.has_events() is True


# ─── ScenarioMultiplier constants ────────────────────────────────────────────
class TestScenarioMultipliers:
    def test_normal_multiplier_one(self):
        assert SCENARIO_MULTIPLIER[ScenarioMode.NORMAL] == 1.0

    def test_crisis_multiplier_greater_than_normal(self):
        assert SCENARIO_MULTIPLIER[ScenarioMode.CRISIS] > SCENARIO_MULTIPLIER[ScenarioMode.NORMAL]

    def test_extreme_multiplier_greatest(self):
        assert (SCENARIO_MULTIPLIER[ScenarioMode.EXTREME_OVERLOAD] >
                SCENARIO_MULTIPLIER[ScenarioMode.CRISIS])

    def test_all_multipliers_positive(self):
        for mode, mult in SCENARIO_MULTIPLIER.items():
            assert mult > 0, f"Multiplier for {mode} should be positive"


# ─── EventEngine construction ────────────────────────────────────────────────
class TestEventEngineConstruction:
    def test_engine_initialises_with_seed_and_mode(self):
        engine = EventEngine(seed=42, scenario_mode=ScenarioMode.NORMAL)
        assert engine.seed == 42
        assert engine.scenario_mode == ScenarioMode.NORMAL

    def test_engine_stores_correct_multiplier(self):
        engine = EventEngine(seed=0, scenario_mode=ScenarioMode.CRISIS)
        assert engine._multiplier == SCENARIO_MULTIPLIER[ScenarioMode.CRISIS]


# ─── Determinism guarantee ────────────────────────────────────────────────────
class TestEventEngineDeterminism:
    def test_same_seed_same_day_same_events(self):
        task = get_task("cross_department_hard")
        engine1 = EventEngine(seed=999, scenario_mode=ScenarioMode.CRISIS)
        engine2 = EventEngine(seed=999, scenario_mode=ScenarioMode.CRISIS)
        for day in range(1, 10):
            e1 = engine1.get_events_for_day(day, task)
            e2 = engine2.get_events_for_day(day, task)
            assert e1 == e2, f"Day {day}: non-deterministic events {e1} vs {e2}"

    def test_different_seeds_can_produce_different_events(self):
        task = get_task("cross_department_hard")
        engine_a = EventEngine(seed=1, scenario_mode=ScenarioMode.CRISIS)
        engine_b = EventEngine(seed=2, scenario_mode=ScenarioMode.CRISIS)
        results_a = [engine_a.get_events_for_day(d, task) for d in range(1, 30)]
        results_b = [engine_b.get_events_for_day(d, task) for d in range(1, 30)]
        # They should differ for at least some days (with high probability)
        assert results_a != results_b

    def test_day_independence(self):
        """Calling day 5 after day 3 gives same result as calling day 5 directly."""
        task = get_task("cross_department_hard")
        engine = EventEngine(seed=42, scenario_mode=ScenarioMode.CRISIS)
        # Call day 3 first, then day 5
        engine.get_events_for_day(3, task)
        day5_after = engine.get_events_for_day(5, task)
        # Fresh engine, only call day 5
        engine2 = EventEngine(seed=42, scenario_mode=ScenarioMode.CRISIS)
        day5_direct = engine2.get_events_for_day(5, task)
        assert day5_after == day5_direct


# ─── Event output format ─────────────────────────────────────────────────────
class TestEventEngineOutput:
    def test_returns_list_of_event_types(self):
        task = get_task("cross_department_hard")
        engine = EventEngine(seed=42, scenario_mode=ScenarioMode.CRISIS)
        events = engine.get_events_for_day(1, task)
        assert isinstance(events, list)
        for e in events:
            assert isinstance(e, EventType)

    def test_no_event_returned_when_none_active(self):
        """Easy task with NO_EVENT allowed — must return [NO_EVENT] not []."""
        task = get_task("district_backlog_easy")
        engine = EventEngine(seed=42, scenario_mode=ScenarioMode.NORMAL)
        events = engine.get_events_for_day(1, task)
        assert len(events) >= 1

    def test_events_only_from_allowed_list(self):
        task = get_task("district_backlog_easy")
        engine = EventEngine(seed=42, scenario_mode=ScenarioMode.NORMAL)
        for day in range(1, 31):
            events = engine.get_events_for_day(day, task)
            for e in events:
                assert e in task.allowed_events or e == EventType.NO_EVENT

    def test_hard_task_can_produce_surge_event(self):
        """With crisis mode + 60 days, a surge event must appear at least once."""
        task = get_task("cross_department_hard")
        engine = EventEngine(seed=999, scenario_mode=ScenarioMode.CRISIS)
        all_events = []
        for day in range(1, 61):
            all_events.extend(engine.get_events_for_day(day, task))
        non_null = [e for e in all_events if e != EventType.NO_EVENT]
        assert len(non_null) > 0, "Crisis mode should produce at least one real event"


# ─── Apply events effects ─────────────────────────────────────────────────────
class TestApplyEvents:
    def _engine(self):
        return EventEngine(seed=42, scenario_mode=ScenarioMode.CRISIS)

    def test_no_event_gives_no_modification(self):
        engine = self._engine()
        task = get_task("district_backlog_easy")
        params = engine.apply_events([EventType.NO_EVENT], task)
        assert params.arrival_multiplier == 1.0
        assert params.officer_reduction == 0

    def test_surge_event_increases_arrival_multiplier(self):
        engine = self._engine()
        task = get_task("cross_department_hard")
        params = engine.apply_events([EventType.SURGE_APPLICATIONS], task)
        assert params.arrival_multiplier > 1.0

    def test_officer_unavailable_reduces_officers(self):
        engine = self._engine()
        task = get_task("cross_department_hard")
        params = engine.apply_events([EventType.OFFICER_UNAVAILABLE], task)
        assert params.officer_reduction >= 1

    def test_doc_rejection_spike_boosts_defect_rate(self):
        engine = self._engine()
        task = get_task("cross_department_hard")
        params = engine.apply_events([EventType.DOCUMENT_REJECTION_SPIKE], task)
        assert params.doc_defect_rate_boost > 0.0

    def test_revenue_db_delay_boosts_system_dependency(self):
        engine = self._engine()
        task = get_task("cross_department_hard")
        params = engine.apply_events([EventType.REVENUE_DB_DELAY], task)
        assert params.system_dependency_boost > 0.0

    def test_sla_escalation_reduces_sla_window(self):
        engine = self._engine()
        task = get_task("cross_department_hard")
        params = engine.apply_events([EventType.SLA_ESCALATION_ORDER], task)
        assert params.sla_window_multiplier <= 1.0

    def test_multiple_events_compound(self):
        engine = self._engine()
        task = get_task("cross_department_hard")
        params = engine.apply_events(
            [EventType.SURGE_APPLICATIONS, EventType.OFFICER_UNAVAILABLE], task
        )
        assert params.arrival_multiplier > 1.0
        assert params.officer_reduction >= 1

    def test_active_events_populated_correctly(self):
        engine = self._engine()
        task = get_task("cross_department_hard")
        params = engine.apply_events([EventType.SURGE_APPLICATIONS], task)
        assert EventType.SURGE_APPLICATIONS in params.active_events

    def test_no_event_gives_no_event_in_active_list(self):
        engine = self._engine()
        task = get_task("district_backlog_easy")
        params = engine.apply_events([EventType.NO_EVENT], task)
        assert params.active_events == [EventType.NO_EVENT]


# ─── Describe events ──────────────────────────────────────────────────────────
class TestDescribeEvents:
    def _engine(self):
        return EventEngine(seed=42, scenario_mode=ScenarioMode.NORMAL)

    def test_no_event_description(self):
        engine = self._engine()
        desc = engine.describe_events([EventType.NO_EVENT])
        assert "No active events" in desc

    def test_surge_description(self):
        engine = self._engine()
        desc = engine.describe_events([EventType.SURGE_APPLICATIONS])
        assert isinstance(desc, str)
        assert len(desc) > 0

    def test_multiple_events_description(self):
        engine = self._engine()
        desc = engine.describe_events([
            EventType.SURGE_APPLICATIONS,
            EventType.OFFICER_UNAVAILABLE,
        ])
        assert ";" in desc  # Two events joined by semicolon
