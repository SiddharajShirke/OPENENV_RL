from app.baselines import run_policy_episode
from app.env import GovWorkflowEnv
from app.graders import grade_episode
from app.models import ActionModel, ActionType

def test_grader_score_range():
    env = GovWorkflowEnv("district_backlog_easy")
    env.reset(seed=123)
    for _ in range(5):
        env.step(ActionModel(action_type=ActionType.ADVANCE_TIME))
    assert 0.0 <= grade_episode(env.state()).score <= 1.0

def test_policy_run_grader_range():
    result = run_policy_episode("mixed_urgency_medium", "urgent_first", seed=22)
    assert 0.0 <= result["score"] <= 1.0