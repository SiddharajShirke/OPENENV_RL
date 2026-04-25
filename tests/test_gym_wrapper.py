"""Tests for the Gymnasium adapter -- validates SB3 contract compliance."""

import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env

from rl.gov_workflow_env import GovWorkflowGymEnv
from rl.feature_builder import OBS_DIM, N_ACTIONS


@pytest.fixture
def env():
    e = GovWorkflowGymEnv(task_id="district_backlog_easy", seed=42)
    yield e


def test_obs_space_shape(env):
    assert env.observation_space.shape == (OBS_DIM,)


def test_action_space_is_discrete(env):
    assert env.action_space.n == N_ACTIONS


def test_reset_returns_numpy_obs(env):
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32


def test_step_returns_gym_contract(env):
    env.reset()
    obs, reward, terminated, truncated, info = env.step(18)
    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_action_masks_returns_bool_array(env):
    env.reset()
    masks = env.action_masks()
    assert isinstance(masks, np.ndarray)
    assert masks.dtype == bool
    assert masks.shape == (N_ACTIONS,)


def test_advance_time_always_valid(env):
    env.reset()
    masks = env.action_masks()
    assert masks[18]


def test_reset_is_deterministic(env):
    obs1, _ = env.reset(seed=42)
    obs2, _ = env.reset(seed=42)
    np.testing.assert_array_equal(obs1, obs2)


def test_obs_values_in_valid_range(env):
    obs, _ = env.reset()
    assert np.all(obs >= -0.01)
    assert np.all(obs <=  1.01)


def test_episode_terminates(env):
    env.reset()
    done, steps = False, 0
    while not done and steps < 1000:
        _, _, terminated, truncated, _ = env.step(18)
        done = terminated or truncated
        steps += 1
    assert done, "Episode did not terminate within 1000 steps"


def test_sb3_check_env_passes():
    env = GovWorkflowGymEnv(task_id="district_backlog_easy", seed=42)
    check_env(env, warn=True)


def test_hard_mask_invalid_action_falls_back_to_advance_time():
    env = GovWorkflowGymEnv(task_id="district_backlog_easy", seed=42, hard_action_mask=True)
    env.reset()
    _, _, _, _, info = env.step(-1)
    assert info["action_mask_applied"] is True
    assert info["executed_action_idx"] == 18


def test_non_advance_streak_forces_advance_time_only():
    env = GovWorkflowGymEnv(task_id="district_backlog_easy", seed=42, max_non_advance_streak=2)
    env.reset(seed=42)
    env.step(18)  # advance one day so backlog appears

    # Two non-advance control actions reach the streak limit.
    env.step(3)
    env.step(2)
    masks = env.action_masks()

    assert masks[18]
    assert int(masks.sum()) == 1
