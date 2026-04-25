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
    # v2 ObservationModel doesn't expose priority_mode directly;
    # verify via the env's internal state and the action explanation
    assert env.priority_mode == PriorityMode.URGENT_FIRST
    assert "urgent_first" in obs.last_action_explanation.lower()

def test_invalid_action_penalized():
    env = GovWorkflowEnv("district_backlog_easy")
    env.reset(seed=123)
    _, reward, _, _, info = env.step(ActionModel(action_type=ActionType.ASSIGN_CAPACITY,
                                                  capacity_assignment={"passport": 99}))
    assert info.invalid_action is True
    assert reward <= 0

def test_reset_is_deterministic():
    obs_a, _ = GovWorkflowEnv("district_backlog_easy").reset(seed=123)
    obs_b, _ = GovWorkflowEnv("district_backlog_easy").reset(seed=123)
    d_a, d_b = obs_a.model_dump(), obs_b.model_dump()
    # episode_id has a random component — strip it
    d_a.pop("episode_id", None); d_b.pop("episode_id", None)
    d_a.pop("last_action_message", None); d_b.pop("last_action_message", None)
    assert d_a == d_b


def test_episode_truncates_on_step_cap_without_advancing_time():
    env = GovWorkflowEnv("district_backlog_easy")
    env.reset(seed=123, options={"max_steps_per_episode": 5})

    done = False
    for _ in range(6):
        _, _, terminated, truncated, _ = env.step(
            ActionModel(
                action_type=ActionType.SET_PRIORITY_MODE,
                priority_mode=PriorityMode.BALANCED,
            )
        )
        done = bool(terminated or truncated)
        if done:
            break

    assert done is True
    assert env.truncated is True
    assert env.total_steps == 5
