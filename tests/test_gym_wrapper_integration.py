import numpy as np

from rl.gov_workflow_env import GovWorkflowGymEnv


def test_gym_wrapper_reset_step_and_core_env_access():
    env = GovWorkflowGymEnv(
        task_id="district_backlog_easy",
        seed=101,
        hard_action_mask=True,
    )

    obs, info = env.reset(seed=101)

    assert isinstance(obs, np.ndarray)
    assert obs.shape == env.observation_space.shape
    assert isinstance(info, dict)
    assert env.core_env is not None

    masks = env.action_masks()
    assert isinstance(masks, np.ndarray)
    assert masks.dtype == bool
    assert masks.shape == (env.action_space.n,)

    valid_actions = np.flatnonzero(masks)
    assert valid_actions.size > 0

    obs2, reward, terminated, truncated, info2 = env.step(int(valid_actions[0]))

    assert isinstance(obs2, np.ndarray)
    assert obs2.shape == env.observation_space.shape
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info2, dict)
    assert "requested_action_idx" in info2
    assert "executed_action_idx" in info2
    assert "action_mask_applied" in info2


def test_gym_wrapper_hard_mask_sanitizes_invalid_action_when_available():
    env = GovWorkflowGymEnv(
        task_id="district_backlog_easy",
        seed=202,
        hard_action_mask=True,
    )
    env.reset(seed=202)
    masks = env.action_masks()

    invalid_actions = np.flatnonzero(~masks)
    if invalid_actions.size == 0:
        return

    invalid_idx = int(invalid_actions[0])
    _, _, _, _, info = env.step(invalid_idx)

    assert info["requested_action_idx"] == invalid_idx
    assert info["executed_action_idx"] != invalid_idx
    assert info["action_mask_applied"] is True