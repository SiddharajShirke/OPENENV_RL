from app.env import GovWorkflowEnv
from app.models import ActionModel, ActionType, PriorityMode

def test_step_advances_day():
    env = GovWorkflowEnv("district_backlog_easy")
    env.reset(seed=123)
    obs, reward, terminated, truncated, info = env.step(ActionModel(action_type=ActionType.ADVANCE_TIME))
    assert obs.day == 1
    assert isinstance(reward, float)

def test_set_priority_mode():
    env = GovWorkflowEnv("district_backlog_easy")
    env.reset(seed=123)
    obs, *_ = env.step(ActionModel(action_type=ActionType.SET_PRIORITY_MODE,
                                    priority_mode=PriorityMode.URGENT_FIRST))
    assert obs.priority_mode == PriorityMode.URGENT_FIRST

def test_invalid_action_penalized():
    env = GovWorkflowEnv("district_backlog_easy")
    env.reset(seed=123)
    _, reward, _, _, info = env.step(ActionModel(action_type=ActionType.ASSIGN_CAPACITY,
                                                  officer_delta=99))
    assert info.invalid_action is True
    assert reward <= 0

def test_reset_is_deterministic():
    obs_a, _ = GovWorkflowEnv("district_backlog_easy").reset(seed=123)
    obs_b, _ = GovWorkflowEnv("district_backlog_easy").reset(seed=123)
    d_a, d_b = obs_a.model_dump(), obs_b.model_dump()
    d_a.pop("last_action_message"); d_b.pop("last_action_message")
    assert d_a == d_b