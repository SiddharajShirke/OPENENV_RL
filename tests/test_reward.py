from app.reward import compute_reward


def test_stability_bonus_only_when_enabled() -> None:
    r_enabled = compute_reward(
        stage_advances=0,
        completions=0,
        active_backlog=0,
        new_sla_breaches=0,
        fairness_gap=0.0,
        fairness_threshold=0.4,
        invalid_action=False,
        idle_capacity=0,
        award_stability_bonus=True,
    )
    r_disabled = compute_reward(
        stage_advances=0,
        completions=0,
        active_backlog=0,
        new_sla_breaches=0,
        fairness_gap=0.0,
        fairness_threshold=0.4,
        invalid_action=False,
        idle_capacity=0,
        award_stability_bonus=False,
    )

    assert r_enabled.stability_bonus > 0.0
    assert r_disabled.stability_bonus == 0.0
    assert r_enabled.total_reward > r_disabled.total_reward
