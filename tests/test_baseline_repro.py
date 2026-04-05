from app.baselines import run_policy_episode

def test_baseline_reproducibility():
    r1 = run_policy_episode("district_backlog_easy", "backlog_clearance", seed=101)
    r2 = run_policy_episode("district_backlog_easy", "backlog_clearance", seed=101)
    assert r1["score"] == r2["score"]
    assert r1["reward_sum"] == r2["reward_sum"]
    assert r1["completed"] == r2["completed"]