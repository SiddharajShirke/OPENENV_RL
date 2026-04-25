"""
test_api.py — Phase 3 HTTP API tests.

Uses httpx.AsyncClient with ASGITransport — fully in-process, zero real
network sockets. pytest-asyncio with asyncio_mode="auto" (set in
pyproject.toml) drives every async test automatically.

Session isolation: each test calls POST /reset independently, gets its own
UUID session_id, and operates only on that session. No cross-test leakage.
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app

BASE = "http://test"


# ── /health ────────────────────────────────────────────────────────────────────

async def test_health_returns_ok() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url=BASE) as c:
        r = await c.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert isinstance(data["active_sessions"], int)
    assert data["active_sessions"] >= 0
    assert set(data["available_tasks"]) == {
        "district_backlog_easy",
        "mixed_urgency_medium",
        "cross_department_hard",
        "district_backlog_easy_extreme",
    }


# ── POST /reset ────────────────────────────────────────────────────────────────

async def test_reset_returns_session_id_and_observation() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url=BASE) as c:
        r = await c.post("/reset", json={"task_id": "district_backlog_easy", "seed": 11})
    assert r.status_code == 200
    data = r.json()
    assert "session_id" in data
    assert len(data["session_id"]) == 36  # UUID4 canonical string length
    obs = data["observation"]
    assert obs["day"] == 0
    assert obs["task_id"] == "district_backlog_easy"
    assert obs["total_backlog"] >= 0


async def test_reset_same_seed_produces_identical_observations() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url=BASE) as c:
        r1 = await c.post("/reset", json={"task_id": "district_backlog_easy", "seed": 11})
        r2 = await c.post("/reset", json={"task_id": "district_backlog_easy", "seed": 11})
    obs1 = r1.json()["observation"]
    obs2 = r2.json()["observation"]
    # Strip volatile fields before comparison
    for obs in (obs1, obs2):
        obs.pop("last_action_message", None)
        obs.pop("episode_id", None)
    assert obs1 == obs2


async def test_reset_medium_task_accepted() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url=BASE) as c:
        r = await c.post("/reset", json={"task_id": "mixed_urgency_medium", "seed": 22})
    assert r.status_code == 200
    assert r.json()["observation"]["task_id"] == "mixed_urgency_medium"


async def test_reset_hard_task_accepted() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url=BASE) as c:
        r = await c.post("/reset", json={"task_id": "cross_department_hard", "seed": 33})
    assert r.status_code == 200
    assert r.json()["observation"]["task_id"] == "cross_department_hard"


async def test_reset_accepts_empty_body_for_validator_compat() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url=BASE) as c:
        r = await c.post("/reset", json={})
    assert r.status_code == 200
    assert "session_id" in r.json()

    async def test_reset_accepts_missing_body_for_validator_compat() -> None:
        async with AsyncClient(transport=ASGITransport(app=app), base_url=BASE) as c:
            r = await c.post("/reset")
        assert r.status_code == 200
        assert "session_id" in r.json()


# ── POST /step ─────────────────────────────────────────────────────────────────

async def test_step_advance_time_moves_day_forward() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url=BASE) as c:
        sid = (await c.post("/reset", json={"task_id": "district_backlog_easy", "seed": 11})).json()["session_id"]
        r = await c.post("/step", json={
            "session_id": sid,
            "action": {"action_type": "advance_time"},
        })
    assert r.status_code == 200
    data = r.json()
    assert data["observation"]["day"] == 1
    assert isinstance(data["reward"], float)
    assert isinstance(data["done"], bool)
    assert data["terminated"] is False
    assert data["truncated"] is False
    assert data["info"]["invalid_action"] is False


async def test_step_set_priority_mode_reflects_in_observation() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url=BASE) as c:
        sid = (await c.post("/reset", json={"task_id": "district_backlog_easy", "seed": 11})).json()["session_id"]
        r = await c.post("/step", json={
            "session_id": sid,
            "action": {"action_type": "set_priority_mode", "priority_mode": "urgent_first"},
        })
    assert r.status_code == 200
    assert "urgent_first" in r.json()["observation"]["last_action_explanation"].lower()


async def test_step_invalid_action_returns_200_with_penalty_not_error() -> None:
    """
    Invalid actions must NOT raise HTTP 4xx/5xx.
    They must return 200 with invalid_action=True and a negative reward.
    """
    async with AsyncClient(transport=ASGITransport(app=app), base_url=BASE) as c:
        sid = (await c.post("/reset", json={"task_id": "district_backlog_easy", "seed": 11})).json()["session_id"]
        r = await c.post("/step", json={
            "session_id": sid,
            "action": {"action_type": "assign_capacity", "officer_delta": 9999},
        })
    assert r.status_code == 200
    data = r.json()
    assert data["info"]["invalid_action"] is True
    assert isinstance(data["info"]["action_explanation"], str)
    assert data["reward"] <= 0


async def test_step_on_ended_episode_returns_409() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url=BASE) as c:
        sid = (await c.post("/reset", json={"task_id": "district_backlog_easy", "seed": 11})).json()["session_id"]
        # Advance past max_days (30) to guarantee truncation
        for _ in range(35):
            await c.post("/step", json={
                "session_id": sid,
                "action": {"action_type": "advance_time"},
            })
        # Next step must be rejected with 409
        r = await c.post("/step", json={
            "session_id": sid,
            "action": {"action_type": "advance_time"},
        })
    assert r.status_code == 409


async def test_step_unknown_session_returns_404() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url=BASE) as c:
        r = await c.post("/step", json={
            "session_id": "00000000-0000-0000-0000-000000000000",
            "action": {"action_type": "advance_time"},
        })
    assert r.status_code == 404


# ── POST /state ────────────────────────────────────────────────────────────────

async def test_state_strips_action_history_by_default() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url=BASE) as c:
        sid = (await c.post("/reset", json={"task_id": "district_backlog_easy", "seed": 11})).json()["session_id"]
        await c.post("/step", json={"session_id": sid, "action": {"action_type": "advance_time"}})
        r = await c.post("/state", json={"session_id": sid, "include_action_history": False})
    assert r.status_code == 200
    assert r.json()["state"]["action_history"] is None


async def test_state_includes_full_action_history_when_requested() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url=BASE) as c:
        sid = (await c.post("/reset", json={"task_id": "district_backlog_easy", "seed": 11})).json()["session_id"]
        for _ in range(3):
            await c.post("/step", json={"session_id": sid, "action": {"action_type": "advance_time"}})
        r = await c.post("/state", json={"session_id": sid, "include_action_history": True})
    assert r.status_code == 200
    data = r.json()
    history = data["state"]["action_history"]
    assert len(history) == 3
    # Each entry must carry the mandatory fields
    for entry in history:
        assert "step" in entry
        assert "day" in entry
        assert "reward" in entry
        assert "invalid" in entry


async def test_state_unknown_session_returns_404() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url=BASE) as c:
        r = await c.post("/state", json={"session_id": "bad-id", "include_action_history": False})
    assert r.status_code == 404


# ── POST /grade ────────────────────────────────────────────────────────────────

async def test_grade_easy_returns_score_in_range_with_correct_grader() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url=BASE) as c:
        sid = (await c.post("/reset", json={"task_id": "district_backlog_easy", "seed": 11})).json()["session_id"]
        for _ in range(5):
            await c.post("/step", json={"session_id": sid, "action": {"action_type": "advance_time"}})
        r = await c.post("/grade", json={"session_id": sid})
    assert r.status_code == 200
    data = r.json()
    assert 0.0 <= data["score"] <= 1.0
    assert data["grader_name"] == "easy"
    assert "completion_rate" in data["metrics"]
    assert "sla_compliance_rate" in data["metrics"]
    assert "idle_efficiency" in data["metrics"]


async def test_grade_medium_task_uses_medium_grader() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url=BASE) as c:
        sid = (await c.post("/reset", json={"task_id": "mixed_urgency_medium", "seed": 22})).json()["session_id"]
        await c.post("/step", json={"session_id": sid, "action": {"action_type": "advance_time"}})
        r = await c.post("/grade", json={"session_id": sid})
    assert r.json()["grader_name"] == "medium"


async def test_grade_hard_task_uses_hard_grader() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url=BASE) as c:
        sid = (await c.post("/reset", json={"task_id": "cross_department_hard", "seed": 33})).json()["session_id"]
        await c.post("/step", json={"session_id": sid, "action": {"action_type": "advance_time"}})
        r = await c.post("/grade", json={"session_id": sid})
    assert r.json()["grader_name"] == "hard"


# ── GET /sessions + DELETE /sessions/{id} ─────────────────────────────────────

async def test_sessions_endpoint_includes_created_session() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url=BASE) as c:
        sid = (await c.post("/reset", json={"task_id": "district_backlog_easy", "seed": 11})).json()["session_id"]
        r = await c.get("/sessions")
    assert r.status_code == 200
    data = r.json()
    assert data["active_sessions"] >= 1
    assert sid in data["session_ids"]


async def test_delete_session_removes_it_and_subsequent_state_returns_404() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url=BASE) as c:
        sid = (await c.post("/reset", json={"task_id": "district_backlog_easy", "seed": 11})).json()["session_id"]
        del_r = await c.delete(f"/sessions/{sid}")
        assert del_r.status_code == 200
        assert del_r.json()["deleted"] == sid
        # Session must be gone
        state_r = await c.post("/state", json={"session_id": sid, "include_action_history": False})
        assert state_r.status_code == 404


async def test_delete_unknown_session_returns_404() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url=BASE) as c:
        r = await c.delete("/sessions/not-a-real-session")
    assert r.status_code == 404


async def test_ui_page_is_served() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url=BASE) as c:
        r = await c.get("/ui")
    assert r.status_code == 200
    assert "text/html" in r.headers.get("content-type", "")


async def test_api_alias_reset_and_autostep_flow() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url=BASE) as c:
        reset_r = await c.post("/api/reset", json={"task_id": "district_backlog_easy", "seed": 11})
        sid = reset_r.json()["session_id"]
        step_r = await c.post(
            "/api/auto_step",
            json={"session_id": sid, "agent_policy": "backlog_clearance"},
        )
    assert step_r.status_code == 200
    data = step_r.json()
    assert data["agent_policy"] == "backlog_clearance"
    assert "action" in data
    assert "observation" in data
    assert isinstance(data["reward"], float)

    async def test_frontend_alias_reset_accepts_missing_body() -> None:
        async with AsyncClient(transport=ASGITransport(app=app), base_url=BASE) as c:
            reset_r = await c.post("/api/reset")
        assert reset_r.status_code == 200
        assert "session_id" in reset_r.json()


async def test_api_benchmark_returns_agent_results() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url=BASE) as c:
        r = await c.post(
            "/api/benchmark",
            json={
                "task_id": "district_backlog_easy",
                "agent_policies": ["urgent_first", "backlog_clearance"],
                "runs": 2,
                "max_steps": 100,
                "seed_base": 500,
            },
        )
    assert r.status_code == 200
    data = r.json()
    assert data["task_id"] == "district_backlog_easy"
    assert data["requested_runs"] == 2
    assert len(data["agent_results"]) == 2
    for agent in data["agent_results"]:
        assert len(agent["runs"]) == 2


async def test_api_benchmark_summary_matches_run_scores_and_is_reproducible() -> None:
    payload = {
        "task_id": "district_backlog_easy",
        "agent_policies": ["urgent_first"],
        "runs": 3,
        "max_steps": 80,
        "seed_base": 777,
    }
    async with AsyncClient(transport=ASGITransport(app=app), base_url=BASE) as c:
        r1 = await c.post("/api/benchmark", json=payload)
        r2 = await c.post("/api/benchmark", json=payload)

    assert r1.status_code == 200
    assert r2.status_code == 200

    a1 = r1.json()["agent_results"][0]
    a2 = r2.json()["agent_results"][0]
    runs1 = a1["runs"]
    scores = [float(row["score"]) for row in runs1]
    expected_avg = sum(scores) / len(scores)

    assert abs(float(a1["average_score"]) - expected_avg) < 1e-9
    assert runs1 == a2["runs"]
    assert float(a1["average_score"]) == float(a2["average_score"])


async def test_api_workflow_components_visible() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url=BASE) as c:
        r = await c.get("/api/workflows/components")
    assert r.status_code == 200
    data = r.json()
    assert "components" in data
    names = {row["component"] for row in data["components"]}
    assert "baseline_openai.py" in names
    assert "inference.py" in names
    assert "openenv-api" in names


async def test_api_rl_models_list_shape() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url=BASE) as c:
        r = await c.get("/api/rl_models")
    assert r.status_code == 200
    data = r.json()
    assert "models" in data
    assert isinstance(data["models"], list)
    assert len(data["models"]) >= 1


async def test_api_rl_run_invalid_model_returns_422() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url=BASE) as c:
        r = await c.post(
            "/api/rl_run",
            json={
                "task_id": "district_backlog_easy",
                "model_path": "results/best_model/does_not_exist.zip",
                "model_type": "maskable",
                "max_steps": 10,
            },
        )
    assert r.status_code == 422


async def test_api_workflow_run_invalid_id_returns_422() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url=BASE) as c:
        r = await c.post(
            "/api/workflows/run",
            json={
                "workflow_id": "not_allowed",
            },
        )
    assert r.status_code == 422


async def test_api_workflow_run_inference_returns_output_fields() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url=BASE) as c:
        r = await c.post(
            "/api/workflows/run",
            json={
                "workflow_id": "inference",
                "max_steps": 1,
                "timeout_seconds": 30,
            },
        )
    assert r.status_code == 200
    data = r.json()
    assert data["workflow_id"] == "inference"
    assert "command" in data
    assert "exit_code" in data
    assert "stdout" in data
    assert "stderr" in data


async def test_api_openenv_compliance_endpoint_returns_items() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url=BASE) as c:
        r = await c.get("/api/openenv_compliance")
    assert r.status_code == 200
    data = r.json()
    assert "items" in data
    assert isinstance(data["items"], list)
    keys = {item["key"] for item in data["items"]}
    assert "api_step_reset_state" in keys
    assert "openenv_yaml" in keys


async def test_api_simulation_live_step_flow_runs_without_500() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url=BASE) as c:
        start = await c.post(
            "/api/simulation/live/start",
            json={
                "task_id": "district_backlog_easy",
                "agent_mode": "llm_inference",
                "max_steps": 10,
                "seed": 11,
            },
        )
        assert start.status_code == 200
        run_id = start.json()["run_id"]
        step = await c.post("/api/simulation/live/step", json={"run_id": run_id})
    assert step.status_code == 200
    payload = step.json()
    assert "run_id" in payload
    assert "total_reward" in payload
    assert isinstance(payload["done"], bool)
