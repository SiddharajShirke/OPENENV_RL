"""
tests/test_story_router.py
Tests for all 7 /training/* endpoints.
Requires: data/training_logs/mixed_urgency_medium_training_log.json
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)
TASK = "mixed_urgency_medium"


def test_list_tasks():
    r = client.get("/training/tasks")
    assert r.status_code == 200
    data = r.json()
    assert "tasks" in data
    assert isinstance(data["tasks"], list)


def test_summary():
    r = client.get(f"/training/summary/{TASK}")
    assert r.status_code == 200
    data = r.json()
    assert data["task_id"] == TASK
    assert "summary" in data
    assert "narrative" in data
    assert "phase_1" in data["narrative"]
    assert "phase_4" in data["narrative"]


def test_curve_full():
    r = client.get(f"/training/curve/{TASK}")
    assert r.status_code == 200
    data = r.json()
    assert "curve" in data
    assert len(data["curve"]) > 0
    ep = data["curve"][0]
    assert "episode" in ep
    assert "reward" in ep
    assert "score" in ep
    assert "phase" in ep


def test_curve_downsample():
    r = client.get(f"/training/curve/{TASK}?downsample=5")
    assert r.status_code == 200
    data = r.json()
    assert data["total_points"] <= 100000


def test_actions():
    r = client.get(f"/training/actions/{TASK}")
    assert r.status_code == 200
    data = r.json()
    assert "checkpoints" in data
    assert len(data["checkpoints"]) == 5
    assert "insight" in data


def test_episode_first():
    r = client.get(f"/training/episode/{TASK}/1")
    assert r.status_code == 200
    data = r.json()
    assert data["episode"] == 1
    assert "reward" in data
    assert "score" in data
    assert "fn1_valid" in data
    assert "fn2_no_halluc" in data
    assert "fn3_env_score" in data
    assert "message" in data
    assert "running_best_reward" in data


def test_episode_last():
    # Get total to know last episode
    summary = client.get(f"/training/summary/{TASK}").json()
    total = summary["total_episodes"]
    r = client.get(f"/training/episode/{TASK}/{total}")
    assert r.status_code == 200


def test_episode_out_of_range():
    r = client.get(f"/training/episode/{TASK}/99999")
    assert r.status_code == 400


def test_comparison():
    r = client.get(f"/training/comparison/{TASK}")
    assert r.status_code == 200
    data = r.json()
    assert "before" in data
    assert "after" in data
    assert "improvement" in data
    assert "verdict" in data["improvement"]
    assert data["before"]["score"] > 0
    assert data["after"]["score"] > 0


def test_missing_task_404():
    r = client.get("/training/summary/nonexistent_task_xyz")
    assert r.status_code == 404


def test_stream_headers():
    # Test SSE endpoint returns correct content-type
    with client.stream("GET", f"/training/stream/{TASK}?delay_ms=0") as r:
        assert r.status_code == 200
        assert "text/event-stream" in r.headers["content-type"]

