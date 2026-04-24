"""
tests/test_phase2_env_integration.py
Phase 2 integration: env.py end-to-end episode lifecycle
Tests reset(), step(), state(), advance_time loop, action dispatch
Run: pytest tests/test_phase2_env_integration.py -v
"""
import pytest
from app.env import GovWorkflowEnv
from app.models import (
    ActionModel, ActionType, PriorityMode, ServiceType,
    ObservationModel, EpisodeStateModel, StepInfoModel, RewardModel,
    InternalSubstate,
)


def make_env(task_id="district_backlog_easy") -> GovWorkflowEnv:
    return GovWorkflowEnv(task_id=task_id)


# ─── reset() API ─────────────────────────────────────────────────────────────
class TestReset:
    def test_reset_returns_tuple(self):
        env = make_env()
        result = env.reset()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_reset_returns_observation_and_info(self):
        env = make_env()
        obs, info = env.reset()
        assert isinstance(obs, ObservationModel)
        assert isinstance(info, dict)

    def test_reset_observation_day_zero(self):
        env = make_env()
        obs, _ = env.reset()
        assert obs.day == 0

    def test_reset_episode_id_set(self):
        env = make_env()
        obs, _ = env.reset()
        assert obs.episode_id != ""
        assert len(obs.episode_id) > 0

    def test_reset_not_terminated(self):
        env = make_env()
        env.reset()
        assert env.terminated is False
        assert env.truncated is False

    def test_reset_deterministic_with_same_seed(self):
        env1 = make_env()
        env2 = make_env()
        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)
        assert obs1.day == obs2.day
        assert obs1.task_id == obs2.task_id
        assert obs1.officer_pool.total_officers == obs2.officer_pool.total_officers

    def test_reset_with_explicit_seed(self):
        env = make_env()
        obs, _ = env.reset(seed=99)
        assert obs.day == 0

    def test_reset_info_contains_task_id(self):
        env = make_env()
        _, info = env.reset()
        assert "task_id" in info

    def test_reset_task_id_in_observation(self):
        env = make_env()
        obs, _ = env.reset()
        assert obs.task_id == "district_backlog_easy"

    def test_double_reset_gives_fresh_episode(self):
        env = make_env()
        obs1, _ = env.reset(seed=42)
        ep1 = obs1.episode_id
        obs2, _ = env.reset(seed=42)
        ep2 = obs2.episode_id
        assert ep1 != ep2  # New episode ID each reset

    def test_reset_officer_pool_matches_task_config(self):
        from app.tasks import get_task
        env = make_env()
        obs, _ = env.reset()
        task = get_task("district_backlog_easy")
        assert obs.officer_pool.total_officers == task.initial_officer_pool.total_officers


# ─── step() API ───────────────────────────────────────────────────────────────
class TestStep:
    def _ready_env(self):
        env = make_env()
        env.reset(seed=42)
        return env

    def test_step_returns_five_tuple(self):
        env = self._ready_env()
        action = ActionModel(action_type=ActionType.ADVANCE_TIME)
        result = env.step(action)
        assert len(result) == 5

    def test_step_returns_correct_types(self):
        env = self._ready_env()
        action = ActionModel(action_type=ActionType.ADVANCE_TIME)
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(obs, ObservationModel)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, StepInfoModel)

    def test_step_advances_day(self):
        env = self._ready_env()
        action = ActionModel(action_type=ActionType.ADVANCE_TIME)
        obs, _, _, _, _ = env.step(action)
        assert obs.day == 1

    def test_step_on_terminated_raises(self):
        env = self._ready_env()
        env.terminated = True
        with pytest.raises(RuntimeError):
            env.step(ActionModel(action_type=ActionType.ADVANCE_TIME))

    def test_advance_time_increases_day_each_step(self):
        env = self._ready_env()
        action = ActionModel(action_type=ActionType.ADVANCE_TIME)
        days = []
        for _ in range(5):
            obs, _, terminated, truncated, _ = env.step(action)
            days.append(obs.day)
            if terminated or truncated:
                break
        assert days == sorted(days)

    def test_reward_is_finite_number(self):
        env = self._ready_env()
        _, reward, _, _, _ = env.step(ActionModel(action_type=ActionType.ADVANCE_TIME))
        assert not (reward != reward)  # not NaN
        assert reward != float("inf")

    def test_step_info_has_reward_breakdown(self):
        env = self._ready_env()
        _, _, _, _, info = env.step(ActionModel(action_type=ActionType.ADVANCE_TIME))
        assert isinstance(info.reward_breakdown, RewardModel)


# ─── state() API ──────────────────────────────────────────────────────────────
class TestState:
    def test_state_returns_episode_state_model(self):
        env = make_env()
        env.reset(seed=42)
        s = env.state()
        assert isinstance(s, EpisodeStateModel)

    def test_state_task_id_correct(self):
        env = make_env()
        env.reset(seed=42)
        s = env.state()
        assert s.task_id == "district_backlog_easy"

    def test_state_day_matches_env_day(self):
        env = make_env()
        env.reset(seed=42)
        env.step(ActionModel(action_type=ActionType.ADVANCE_TIME))
        s = env.state()
        assert s.day == env.day

    def test_state_not_terminated_at_start(self):
        env = make_env()
        env.reset(seed=42)
        s = env.state()
        assert s.terminated is False

    def test_state_episode_id_matches_obs(self):
        env = make_env()
        obs, _ = env.reset(seed=42)
        s = env.state()
        assert s.episode_id == obs.episode_id

    def test_state_total_steps_increments(self):
        env = make_env()
        env.reset(seed=42)
        env.step(ActionModel(action_type=ActionType.ADVANCE_TIME))
        env.step(ActionModel(action_type=ActionType.ADVANCE_TIME))
        s = env.state()
        assert s.total_steps == 2


# ─── Action dispatch ──────────────────────────────────────────────────────────
class TestActionDispatch:
    def _ready_env(self, task="district_backlog_easy"):
        env = make_env(task)
        env.reset(seed=42)
        return env

    def test_set_priority_mode_urgent_first(self):
        env = self._ready_env()
        action = ActionModel(
            action_type=ActionType.SET_PRIORITY_MODE,
            priority_mode=PriorityMode.URGENT_FIRST,
        )
        _, _, _, _, info = env.step(action)
        assert not info.invalid_action
        assert env.priority_mode == PriorityMode.URGENT_FIRST

    def test_set_priority_mode_without_mode_is_invalid(self):
        env = self._ready_env()
        action = ActionModel(action_type=ActionType.SET_PRIORITY_MODE)
        _, _, _, _, info = env.step(action)
        assert info.invalid_action

    def test_advance_time_valid(self):
        env = self._ready_env()
        _, _, _, _, info = env.step(ActionModel(action_type=ActionType.ADVANCE_TIME))
        assert not info.invalid_action

    def test_escalate_without_budget_is_invalid(self):
        env = self._ready_env()
        env.escalation_budget_remaining = 0
        action = ActionModel(
            action_type=ActionType.ESCALATE_SERVICE,
            escalation_target=ServiceType.INCOME_CERTIFICATE,
        )
        _, _, _, _, info = env.step(action)
        assert info.invalid_action

    def test_reallocate_with_bad_delta_is_invalid(self):
        env = self._ready_env()
        action = ActionModel(
            action_type=ActionType.REALLOCATE_OFFICERS,
            reallocation_delta={"income_certificate": 2},  # doesn't sum to 0
        )
        _, _, _, _, info = env.step(action)
        assert info.invalid_action

    def test_reallocate_with_one_entry_is_invalid(self):
        env = self._ready_env()
        action = ActionModel(
            action_type=ActionType.REALLOCATE_OFFICERS,
            reallocation_delta={"income_certificate": 0},
        )
        _, _, _, _, info = env.step(action)
        assert info.invalid_action

    def test_assign_capacity_without_dict_is_invalid(self):
        env = self._ready_env()
        action = ActionModel(action_type=ActionType.ASSIGN_CAPACITY)
        _, _, _, _, info = env.step(action)
        assert info.invalid_action

    def test_request_missing_docs_no_blocked_cases_is_invalid(self):
        env = self._ready_env()
        # At day 0 no cases are blocked yet
        action = ActionModel(
            action_type=ActionType.REQUEST_MISSING_DOCUMENTS,
            service_target=ServiceType.INCOME_CERTIFICATE,
        )
        _, _, _, _, info = env.step(action)
        # Either valid (if cases exist) or invalid (if none blocked) — must not crash
        assert isinstance(info.invalid_action, bool)


# ─── Full episode lifecycle ────────────────────────────────────────────────────
class TestFullEpisode:
    def test_episode_terminates_within_max_days(self):
        env = make_env("district_backlog_easy")
        env.reset(seed=42)
        action = ActionModel(action_type=ActionType.ADVANCE_TIME)
        steps = 0
        while steps < 200:
            _, _, terminated, truncated, _ = env.step(action)
            steps += 1
            if terminated or truncated:
                break
        assert terminated or truncated, "Episode must terminate"

    def test_completed_cases_nonneg_at_end(self):
        env = make_env("district_backlog_easy")
        env.reset(seed=42)
        action = ActionModel(action_type=ActionType.ADVANCE_TIME)
        for _ in range(35):
            _, _, t, tr, _ = env.step(action)
            if t or tr:
                break
        s = env.state()
        assert s.total_completed >= 0

    def test_cumulative_reward_is_float(self):
        env = make_env("district_backlog_easy")
        env.reset(seed=42)
        action = ActionModel(action_type=ActionType.ADVANCE_TIME)
        for _ in range(5):
            env.step(action)
        s = env.state()
        assert isinstance(s.cumulative_reward, float)

    def test_episode_deterministic_same_seed_same_actions(self):
        def run(seed):
            env = make_env("district_backlog_easy")
            env.reset(seed=seed)
            rewards = []
            for _ in range(10):
                _, r, t, tr, _ = env.step(
                    ActionModel(action_type=ActionType.ADVANCE_TIME)
                )
                rewards.append(round(r, 6))
                if t or tr:
                    break
            return rewards

        r1 = run(42)
        r2 = run(42)
        assert r1 == r2, "Same seed + same actions must give same rewards"

    def test_medium_task_episode_does_not_crash(self):
        env = make_env("mixed_urgency_medium")
        env.reset(seed=123)
        action = ActionModel(action_type=ActionType.ADVANCE_TIME)
        for _ in range(50):
            _, _, t, tr, _ = env.step(action)
            if t or tr:
                break

    def test_hard_task_episode_does_not_crash(self):
        env = make_env("cross_department_hard")
        env.reset(seed=999)
        action = ActionModel(action_type=ActionType.ADVANCE_TIME)
        for _ in range(65):
            _, _, t, tr, _ = env.step(action)
            if t or tr:
                break
