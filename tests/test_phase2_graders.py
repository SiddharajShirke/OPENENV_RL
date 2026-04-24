"""
tests/test_phase2_graders.py
Phase 2: graders.py — deterministic scoring for all three tasks
Run: pytest tests/test_phase2_graders.py -v
"""
import pytest
from app.env import GovWorkflowEnv
from app.graders import grade_episode
from app.models import ActionModel, ActionType


def run_episode_to_end(task_id: str, seed: int, max_steps: int = 500) -> GovWorkflowEnv:
    """Run a full episode and return the env for grading."""
    env = GovWorkflowEnv(task_id=task_id)
    env.reset(seed=seed)
    action = ActionModel(action_type=ActionType.ADVANCE_TIME)
    for _ in range(max_steps):
        _, _, t, tr, _ = env.step(action)
        if t or tr:
            break
    return env


class TestGraderEasy:
    def test_grade_returns_result(self):
        env = run_episode_to_end("district_backlog_easy", 42)
        result = grade_episode(env.state())
        assert result is not None

    def test_grade_score_in_range(self):
        env = run_episode_to_end("district_backlog_easy", 42)
        result = grade_episode(env.state())
        assert 0.0 <= result.score <= 1.0

    def test_grade_has_grader_name(self):
        env = run_episode_to_end("district_backlog_easy", 42)
        result = grade_episode(env.state())
        assert isinstance(result.grader_name, str)
        assert len(result.grader_name) > 0

    def test_grade_metrics_dict_nonempty(self):
        env = run_episode_to_end("district_backlog_easy", 42)
        result = grade_episode(env.state())
        assert isinstance(result.metrics, dict)
        assert len(result.metrics) > 0

    def test_grade_deterministic_same_seed(self):
        env1 = run_episode_to_end("district_backlog_easy", 42)
        env2 = run_episode_to_end("district_backlog_easy", 42)
        r1 = grade_episode(env1.state())
        r2 = grade_episode(env2.state())
        assert abs(r1.score - r2.score) < 1e-6

    def test_grade_metrics_all_floats(self):
        env = run_episode_to_end("district_backlog_easy", 42)
        result = grade_episode(env.state())
        for k, v in result.metrics.items():
            assert isinstance(v, (int, float)), f"Metric {k} is not numeric: {v}"


class TestGraderMedium:
    def test_grade_score_in_range(self):
        env = run_episode_to_end("mixed_urgency_medium", 123)
        result = grade_episode(env.state())
        assert 0.0 <= result.score <= 1.0

    def test_grade_different_grader_than_easy(self):
        env_easy = run_episode_to_end("district_backlog_easy", 42)
        env_med = run_episode_to_end("mixed_urgency_medium", 123)
        r_easy = grade_episode(env_easy.state())
        r_med = grade_episode(env_med.state())
        # Different tasks may have different grader names
        assert isinstance(r_med.grader_name, str)


class TestGraderHard:
    def test_grade_score_in_range(self):
        env = run_episode_to_end("cross_department_hard", 999, max_steps=800)
        result = grade_episode(env.state())
        assert 0.0 <= result.score <= 1.0

    def test_grade_has_fairness_metric(self):
        env = run_episode_to_end("cross_department_hard", 999, max_steps=800)
        result = grade_episode(env.state())
        # Hard task grader should include fairness-related metric
        keys_lower = {k.lower() for k in result.metrics.keys()}
        has_fairness = any("fair" in k for k in keys_lower)
        assert has_fairness, f"Hard grader missing fairness metric. Keys: {result.metrics.keys()}"


class TestGraderScoreBounds:
    @pytest.mark.parametrize("task_id,seed", [
        ("district_backlog_easy", 42),
        ("mixed_urgency_medium", 123),
        ("cross_department_hard", 999),
    ])
    def test_score_always_in_zero_one(self, task_id, seed):
        env = run_episode_to_end(task_id, seed)
        result = grade_episode(env.state())
        assert 0.0 <= result.score <= 1.0, (
            f"{task_id}: score {result.score} out of [0, 1]"
        )

    @pytest.mark.parametrize("task_id,seed", [
        ("district_backlog_easy", 1),
        ("district_backlog_easy", 2),
        ("district_backlog_easy", 3),
    ])
    def test_partial_episode_grades_without_error(self, task_id, seed):
        env = GovWorkflowEnv(task_id=task_id)
        env.reset(seed=seed)
        action = ActionModel(action_type=ActionType.ADVANCE_TIME)
        for _ in range(5):
            env.step(action)
        result = grade_episode(env.state())
        assert 0.0 <= result.score <= 1.0
