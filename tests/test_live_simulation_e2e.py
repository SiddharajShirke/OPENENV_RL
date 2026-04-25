from app.engine import LiveSimulationSession, SimulationAgentMode, run_simulation


def test_run_simulation_baseline_policy_end_to_end():
    result = run_simulation(
        task_id="district_backlog_easy",
        agent_mode=SimulationAgentMode.BASELINE_POLICY,
        max_steps=12,
        seed=123,
        policy_name="backlog_clearance",
    )

    assert result.task_id == "district_backlog_easy"
    assert result.agent_mode == SimulationAgentMode.BASELINE_POLICY
    assert result.seed == 123
    assert isinstance(result.total_reward, float)
    assert 0.0 <= result.score <= 1.0
    assert isinstance(result.summary, dict)
    assert isinstance(result.trace, list)
    assert len(result.trace) > 0


def test_live_session_step_once_smoke():
    session = LiveSimulationSession(
        task_id="district_backlog_easy",
        agent_mode=SimulationAgentMode.BASELINE_POLICY,
        max_steps=5,
        seed=7,
        policy_name="backlog_clearance",
    )
    try:
        row, log_line, finished = session.step_once()
        assert isinstance(row, dict)
        assert isinstance(log_line, str)
        assert "[STEP]" in log_line
        assert "reward" in row
        assert isinstance(finished, bool)
    finally:
        session.close()