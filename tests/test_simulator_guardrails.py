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
        assert fixed.action_type == ActionType.ADVANCE_TIME
        assert note is not None
    finally:
        session.close()
