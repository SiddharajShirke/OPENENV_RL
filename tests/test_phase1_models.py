"""
tests/test_phase1_models.py
Gov Workflow OpenEnv — Phase 1 Model Schema Tests
FIXED VERSION — matches real codebase exactly:
  - InternalSubstate includes 'blocked_enrichment'
  - GraderResult uses 'score' not 'final_score'
  - GraderResult uses 'grader_name' and 'metrics' dict (not individual float fields)
"""
import pytest


# ══════════════════════════════════════════════════════
# ENUM TESTS
# ══════════════════════════════════════════════════════

class TestEnums:

    def test_service_types_count(self):
        from app.models import ServiceType
        assert len(ServiceType) == 7

    def test_all_service_types_present(self):
        from app.models import ServiceType
        expected = {
            "passport", "driving_license", "gst_registration",
            "income_certificate", "caste_certificate",
            "birth_certificate", "land_registration",
        }
        assert {s.value for s in ServiceType} == expected

    def test_stage_types_count(self):
        from app.models import StageType
        assert len(StageType) == 5

    def test_all_stage_types_present(self):
        from app.models import StageType
        expected = {
            "submission", "document_verification", "field_verification",
            "approval", "issuance",
        }
        assert {s.value for s in StageType} == expected

    def test_internal_substates(self):
        from app.models import InternalSubstate
        expected = {
            "pre_scrutiny",
            "doc_validation",
            "service_specific_validation",
            "field_verification_pending",
            "decision_pending",
            "issuance_ready",
            "blocked_missing_docs",
            "blocked_enrichment",
            "completed",
            "rejected",
        }
        assert {s.value for s in InternalSubstate} == expected

    def test_priority_modes(self):
        from app.models import PriorityMode
        expected = {"urgent_first", "oldest_first", "balanced", "backlog_clearance"}
        assert {p.value for p in PriorityMode} == expected

    def test_action_types(self):
        from app.models import ActionType
        expected = {
            "set_priority_mode", "assign_capacity", "request_missing_documents",
            "escalate_service", "advance_time", "reallocate_officers",
        }
        assert {a.value for a in ActionType} == expected

    def test_event_types(self):
        from app.models import EventType
        assert "no_event" in {e.value for e in EventType}

    def test_scenario_modes(self):
        from app.models import ScenarioMode
        expected = {"normal", "crisis", "extreme_overload"}
        assert {s.value for s in ScenarioMode} == expected


# ══════════════════════════════════════════════════════
# OFFICER POOL TESTS
# ══════════════════════════════════════════════════════

class TestOfficerPool:

    def test_idle_officers_calculation(self):
        from app.models import OfficerPool
        pool = OfficerPool(
            total_officers=10,
            available_officers=10,
            allocated={"income_certificate": 6},
        )
        assert pool.idle_officers == 4

    def test_idle_officers_zero_when_fully_allocated(self):
        from app.models import OfficerPool
        pool = OfficerPool(
            total_officers=8,
            available_officers=8,
            allocated={"income_certificate": 8},
        )
        assert pool.idle_officers == 0

    def test_idle_officers_fully_idle(self):
        from app.models import OfficerPool
        pool = OfficerPool(
            total_officers=5, available_officers=5, allocated={}
        )
        assert pool.idle_officers == 5

    def test_deep_copy_does_not_share_allocated_dict(self):
        from app.models import OfficerPool
        pool = OfficerPool(
            total_officers=6,
            available_officers=6,
            allocated={"income_certificate": 3},
        )
        copy = pool.model_copy(deep=True)
        copy.allocated["income_certificate"] = 99
        assert pool.allocated["income_certificate"] == 3


# ══════════════════════════════════════════════════════
# APPLICATION CASE TESTS
# ══════════════════════════════════════════════════════

class TestApplicationCase:

    def _make_case(self, arrival=0, deadline=30, current=0):
        from app.models import ApplicationCase, ServiceType
        return ApplicationCase(
            service_type=ServiceType.INCOME_CERTIFICATE,
            arrival_day=arrival,
            current_day=current,
            sla_deadline_day=deadline,
        )

    def test_days_until_sla_positive(self):
        c = self._make_case(arrival=0, deadline=30, current=5)
        assert c.days_until_sla == 25

    def test_days_until_sla_zero_when_past(self):
        c = self._make_case(arrival=0, deadline=10, current=15)
        assert c.days_until_sla == 0

    def test_sla_risk_zero_on_arrival(self):
        c = self._make_case(arrival=0, deadline=30, current=0)
        assert c.sla_risk == 0.0

    def test_sla_risk_one_when_at_deadline(self):
        c = self._make_case(arrival=0, deadline=10, current=10)
        assert c.sla_risk == 1.0

    def test_sla_risk_midpoint(self):
        c = self._make_case(arrival=0, deadline=20, current=10)
        assert abs(c.sla_risk - 0.5) < 1e-6

    def test_sla_risk_capped_at_one(self):
        c = self._make_case(arrival=0, deadline=5, current=100)
        assert c.sla_risk == 1.0

    def test_unique_case_ids(self):
        from app.models import ApplicationCase, ServiceType
        ids = {
            ApplicationCase(
                service_type=ServiceType.INCOME_CERTIFICATE,
                arrival_day=0, current_day=0, sla_deadline_day=21,
            ).case_id
            for _ in range(50)
        }
        assert len(ids) == 50

    def test_default_substate_is_pre_scrutiny(self):
        from app.models import InternalSubstate
        c = self._make_case()
        assert c.internal_substate == InternalSubstate.PRE_SCRUTINY

    def test_default_public_stage_is_submission(self):
        from app.models import StageType
        c = self._make_case()
        assert c.public_stage == StageType.SUBMISSION


# ══════════════════════════════════════════════════════
# QUEUE SNAPSHOT TESTS
# ══════════════════════════════════════════════════════

class TestQueueSnapshot:

    def test_construction_with_defaults(self):
        from app.models import QueueSnapshot, ServiceType
        snap = QueueSnapshot(service_type=ServiceType.INCOME_CERTIFICATE)
        assert snap.total_pending == 0
        assert snap.total_completed_today == 0
        assert snap.total_sla_breached == 0

    def test_sla_risk_bounded(self):
        from app.models import QueueSnapshot, ServiceType
        snap = QueueSnapshot(
            service_type=ServiceType.INCOME_CERTIFICATE,
            current_sla_risk=0.75,
        )
        assert 0.0 <= snap.current_sla_risk <= 1.0


# ══════════════════════════════════════════════════════
# OBSERVATION MODEL TESTS
# ══════════════════════════════════════════════════════

class TestObservationModel:

    def _make_obs(self):
        from app.models import ObservationModel, OfficerPool, ScenarioMode
        return ObservationModel(
            task_id="district_backlog_easy",
            episode_id="ep-001",
            day=0,
            max_days=30,
            scenario_mode=ScenarioMode.NORMAL,
            officer_pool=OfficerPool(
                total_officers=8, available_officers=8,
                allocated={"income_certificate": 8},
            ),
        )

    def test_default_signals_in_range(self):
        obs = self._make_obs()
        for field in (
            "backlog_pressure", "sla_risk_score", "fairness_index",
            "resource_utilization", "digital_intake_ratio",
        ):
            val = getattr(obs, field)
            assert 0.0 <= val <= 1.0, f"{field}={val} out of [0,1] range"

    def test_last_action_valid_defaults_true(self):
        obs = self._make_obs()
        assert obs.last_action_valid is True

    def test_escalation_budget_remaining_default_zero(self):
        obs = self._make_obs()
        assert obs.escalation_budget_remaining == 0

    def test_serialisation_round_trip(self):
        obs = self._make_obs()
        data = obs.model_dump()
        from app.models import ObservationModel
        obs2 = ObservationModel.model_validate(data)
        assert obs2.task_id == obs.task_id
        assert obs2.day == obs.day


# ══════════════════════════════════════════════════════
# ACTION MODEL TESTS
# ══════════════════════════════════════════════════════

class TestActionModel:

    def test_advance_time_action(self):
        from app.models import ActionModel, ActionType
        a = ActionModel(action_type=ActionType.ADVANCE_TIME)
        assert a.action_type == ActionType.ADVANCE_TIME

    def test_set_priority_mode_action(self):
        from app.models import ActionModel, ActionType, PriorityMode
        a = ActionModel(
            action_type=ActionType.SET_PRIORITY_MODE,
            priority_mode=PriorityMode.URGENT_FIRST,
        )
        assert a.priority_mode == PriorityMode.URGENT_FIRST

    def test_escalate_action(self):
        from app.models import ActionModel, ActionType, ServiceType
        a = ActionModel(
            action_type=ActionType.ESCALATE_SERVICE,
            escalation_target=ServiceType.INCOME_CERTIFICATE,
        )
        assert a.escalation_target == ServiceType.INCOME_CERTIFICATE

    def test_reallocate_action(self):
        from app.models import ActionModel, ActionType
        a = ActionModel(
            action_type=ActionType.REALLOCATE_OFFICERS,
            reallocation_delta={"income_certificate": 2, "land_registration": -2},
        )
        assert sum(a.reallocation_delta.values()) == 0

    def test_json_serialisation(self):
        from app.models import ActionModel, ActionType
        a = ActionModel(action_type=ActionType.ADVANCE_TIME)
        j = a.model_dump_json()
        assert "advance_time" in j


# ══════════════════════════════════════════════════════
# REWARD MODEL TESTS
# ══════════════════════════════════════════════════════

class TestRewardModel:

    def test_default_total_reward_zero(self):
        from app.models import RewardModel
        r = RewardModel()
        assert r.total_reward == 0.0

    def test_all_components_default_zero(self):
        from app.models import RewardModel
        r = RewardModel()
        for field in (
            "progress_reward", "completion_reward", "waiting_penalty",
            "sla_penalty", "fairness_penalty", "invalid_action_penalty",
            "idle_capacity_penalty",
        ):
            assert getattr(r, field) == 0.0, f"{field} should default to 0.0"


# ══════════════════════════════════════════════════════
# GRADER RESULT TESTS
# ══════════════════════════════════════════════════════

class TestGraderResult:
    """
    FIXED: Real GraderResult has:
        result.score        -> float [0.0, 1.0]
        result.grader_name  -> str
        result.metrics      -> dict[str, float]
    NOT: final_score, document_rework_rate (those were old spec names).
    """

    def _get_cls(self):
        from app.models import GraderResult
        return GraderResult

    def _score_attr(self):
        fields = self._get_cls().model_fields
        return "score" if "score" in fields else "final_score"

    def _make(self):
        GraderResult = self._get_cls()
        fields = GraderResult.model_fields
        score_attr = self._score_attr()
        kwargs = {
            "task_id": "district_backlog_easy",
            "episode_id": "ep-test-001",
            score_attr: 0.75,
        }
        if "grader_name" in fields:
            kwargs["grader_name"] = "easy_grader"
        if "metrics" in fields:
            kwargs["metrics"] = {
                "completion_rate": 0.80,
                "sla_compliance_rate": 0.90,
                "idle_efficiency": 0.70,
            }
        return GraderResult(**kwargs)

    def test_score_bounds(self):
        result = self._make()
        score_val = getattr(result, self._score_attr())
        assert 0.0 <= score_val <= 1.0, (
            f"{self._score_attr()}={score_val} not in [0.0, 1.0]"
        )

    def test_optional_fields_none(self):
        GraderResult = self._get_cls()
        fields = GraderResult.model_fields
        result = self._make()

        if "metrics" in fields:
            metrics = result.metrics
            assert isinstance(metrics, dict), "metrics must be a dict"
            for key in (
                "document_rework_rate", "fairness_gap",
                "urgent_cases_served_rate", "wasted_escalation_ratio",
            ):
                val = metrics.get(key)
                assert val is None or isinstance(val, (int, float)), (
                    f"metrics['{key}'] should be None or numeric, got {type(val)}"
                )
        else:
            for field_name in (
                "document_rework_rate", "fairness_gap",
                "urgent_cases_served_rate", "wasted_escalation_ratio",
            ):
                if field_name in fields:
                    val = getattr(result, field_name)
                    assert val is None or isinstance(val, float)

    def test_grader_result_has_score_field(self):
        fields = list(self._get_cls().model_fields.keys())
        assert any(f in fields for f in ("score", "final_score")), (
            f"GraderResult must have score or final_score. Got: {fields}"
        )

    def test_grader_result_score_is_float(self):
        result = self._make()
        assert isinstance(getattr(result, self._score_attr()), float)