from app.models import ActionModel, ActionType
from app.simulator import LiveSimulationSession, _repair_action_for_observation


def test_reallocate_payload_is_repaired_to_valid_shape() -> None:
    session = LiveSimulationSession(
        task_id="district_backlog_easy",
        agent_mode="baseline_policy",
        max_steps=5,
        seed=42,
    )
    try:
        raw = ActionModel(action_type=ActionType.REALLOCATE_OFFICERS, officer_delta=1)
        fixed, note = _repair_action_for_observation(raw, session.obs)
        assert fixed.action_type == ActionType.REALLOCATE_OFFICERS
        assert fixed.service is not None
        assert fixed.target_service is not None
        assert fixed.service != fixed.target_service
        assert fixed.officer_delta > 0
        assert note is not None
    finally:
        session.close()


def test_assign_capacity_switches_to_advance_time_if_no_reserve() -> None:
    session = LiveSimulationSession(
        task_id="district_backlog_easy",
        agent_mode="baseline_policy",
        max_steps=5,
        seed=42,
    )
    try:
        session.obs.officer_pool.reserve_officers = 0
        raw = ActionModel(action_type=ActionType.ASSIGN_CAPACITY, officer_delta=2)
        fixed, note = _repair_action_for_observation(raw, session.obs)
        assert fixed.action_type in {
            ActionType.ADVANCE_TIME,
            ActionType.REQUEST_MISSING_DOCUMENTS,
            ActionType.REALLOCATE_OFFICERS,
            ActionType.ESCALATE_SERVICE,
        }
        assert note is not None
    finally:
        session.close()


def test_llm_mode_enforces_recommended_min_steps_for_hard_task() -> None:
    session = LiveSimulationSession(
        task_id="cross_department_hard",
        agent_mode="llm_inference",
        max_steps=20,
        seed=42,
    )
    try:
        assert session.max_steps >= 70
    finally:
        session.close()


def test_llm_step_core_handles_none_action_without_crash() -> None:
    session = LiveSimulationSession(
        task_id="district_backlog_easy",
        agent_mode="llm_inference",
        max_steps=10,
        seed=11,
    )
    try:
        # Simulate a malformed llm policy output.
        session.policy = lambda _obs: (None, {"decision_source": "llm", "provider": "test", "model_used": "bad"})
        row, _log, done = session.step_once()
        assert isinstance(row, dict)
        assert row["action_type"] in {a.value for a in ActionType}
        assert isinstance(done, bool)
    finally:
        session.close()
