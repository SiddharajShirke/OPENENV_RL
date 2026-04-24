"""
tests/conftest.py
Shared fixtures for all test modules.
"""
import pytest
from app.env import GovWorkflowEnv
from app.models import ActionModel, ActionType


@pytest.fixture
def easy_env():
    """Fresh GovWorkflowEnv for district_backlog_easy, seed=42."""
    env = GovWorkflowEnv(task_id="district_backlog_easy")
    env.reset(seed=42)
    return env


@pytest.fixture
def medium_env():
    env = GovWorkflowEnv(task_id="mixed_urgency_medium")
    env.reset(seed=123)
    return env


@pytest.fixture
def hard_env():
    env = GovWorkflowEnv(task_id="cross_department_hard")
    env.reset(seed=999)
    return env


@pytest.fixture
def advance_action():
    return ActionModel(action_type=ActionType.ADVANCE_TIME)


@pytest.fixture
def run_episode(easy_env, advance_action):
    """Run easy_env for 10 steps, return list of rewards."""
    rewards = []
    for _ in range(10):
        _, r, t, tr, _ = easy_env.step(advance_action)
        rewards.append(r)
        if t or tr:
            break
    return rewards
