"""Phase 3 tests for recurrent evaluation helpers."""

from __future__ import annotations

import numpy as np

import rl.evaluate as eval_mod
from rl.evaluate import TaskEvalResult, compare_recurrent_vs_flat, predict_recurrent_action


class _FakeRecurrentModel:
    def __init__(self):
        self.seen_states = []

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        self.seen_states.append(state)
        if episode_start is not None and bool(np.asarray(episode_start).item()):
            next_state = 0
        elif state is None:
            next_state = 0
        else:
            next_state = int(state) + 1
        return np.array([18]), next_state


def test_recurrent_policy_hidden_state_persists_across_steps() -> None:
    model = _FakeRecurrentModel()
    obs = np.zeros(4, dtype=np.float32)
    masks = np.array([True] * 28)

    action_1, state_1 = predict_recurrent_action(
        model=model,
        obs=obs,
        lstm_state=None,
        episode_start=np.array([False], dtype=bool),
        masks=masks,
    )
    action_2, state_2 = predict_recurrent_action(
        model=model,
        obs=obs,
        lstm_state=state_1,
        episode_start=np.array([False], dtype=bool),
        masks=masks,
    )

    assert action_1 == 18
    assert action_2 == 18
    assert state_1 == 0
    assert state_2 == 1
    assert model.seen_states == [None, 0]


def test_lstm_reset_on_episode_boundary() -> None:
    model = _FakeRecurrentModel()
    obs = np.zeros(4, dtype=np.float32)
    masks = np.array([True] * 28)

    _, state_1 = predict_recurrent_action(
        model=model,
        obs=obs,
        lstm_state=5,
        episode_start=np.array([False], dtype=bool),
        masks=masks,
    )
    _, state_2 = predict_recurrent_action(
        model=model,
        obs=obs,
        lstm_state=state_1,
        episode_start=np.array([True], dtype=bool),
        masks=masks,
    )

    assert state_1 == 6
    assert state_2 == 0


def test_score_recurrent_geq_flat_ppo_on_medium(monkeypatch) -> None:
    def _fake_evaluate_model(model_path, task_ids, n_episodes, verbose, model_type):
        assert task_ids == ["mixed_urgency_medium"]
        if model_type == "maskable":
            score = 0.60
        else:
            score = 0.63
        return [
            TaskEvalResult(
                task_id="mixed_urgency_medium",
                seed=22,
                grader_score=score,
                total_reward=200.0,
                total_steps=50,
                total_completed=40,
                total_sla_breaches=20,
                fairness_gap=0.1,
            )
        ]

    monkeypatch.setattr(eval_mod, "evaluate_model", _fake_evaluate_model)

    comparison = compare_recurrent_vs_flat(
        flat_model_path="flat.zip",
        recurrent_model_path="recurrent.zip",
        task_id="mixed_urgency_medium",
        n_episodes=3,
    )

    assert comparison["recurrent"] >= comparison["flat"]


def test_invalid_action_prefers_advance_time_fallback() -> None:
    model = _FakeRecurrentModel()
    obs = np.zeros(4, dtype=np.float32)
    masks = np.array([False] * 28)
    masks[18] = True

    action, _ = predict_recurrent_action(
        model=model,
        obs=obs,
        lstm_state=None,
        episode_start=np.array([False], dtype=bool),
        masks=masks,
    )
    assert action == 18
