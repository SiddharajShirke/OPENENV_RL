"""
End-to-end API suite for the full endpoint contract.

This suite focuses on:
1) endpoint availability
2) cross-endpoint data flow
3) session lifecycle correctness
4) simulation stream behavior
5) RL endpoint guardrails
"""

from __future__ import annotations

from httpx import ASGITransport, AsyncClient

from app.main import app
from rl.feature_builder import N_ACTIONS

BASE_URL = "http://test"

REQUIRED_PATHS = {
    "/health",
    "/reset",
    "/step",
    "/state",
    "/simulate",
    "/simulate/{session_id}/snapshot",
    "/grade",
    "/tasks",
    "/tasks/{task_id}",
    "/action-masks",
    "/rl/run",
    "/rl/models",
    "/simulate/{session_id}/cancel",
    "/simulate/{session_id}/trace",
    "/actions/schema",
    "/metrics",
}
async def test_openapi_contains_all_required_endpoints() -> None:
    paths = set(app.openapi().get("paths", {}).keys())
    assert REQUIRED_PATHS.issubset(paths), f"Missing paths: {sorted(REQUIRED_PATHS - paths)}"


async def test_health_tasks_metrics_and_schema_consistency() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url=BASE_URL) as c:
        health = await c.get("/health")
        tasks = await c.get("/tasks")
        metrics = await c.get("/metrics")
        schema = await c.get("/actions/schema")

    assert health.status_code == 200
    h = health.json()
    assert h["status"] in {"ok", "degraded"}
    assert h["version"] == "2.0.0"
    assert h["phase"] == "3_rl_training"

    assert tasks.status_code == 200
    task_rows = tasks.json()
    assert isinstance(task_rows, list)
    assert len(task_rows) == 3
    task_ids = {row["task_id"] for row in task_rows}
    assert task_ids == {
        "district_backlog_easy",
        "mixed_urgency_medium",
        "cross_department_hard",
    }

    assert metrics.status_code == 200
    m = metrics.json()
    assert m["version"] == "2.0.0"
    assert m["phase"] == "3_rl_training"
    assert m["total_tasks"] == 3
    assert set(m["tasks_available"]) == task_ids

    assert schema.status_code == 200
    s = schema.json()
    assert s["total_action_types"] == 6
    assert len(s["actions"]) == 6


async def test_per_task_details_and_unknown_task_404() -> None:
    known = [
        "district_backlog_easy",
        "mixed_urgency_medium",
        "cross_department_hard",
    ]
    async with AsyncClient(transport=ASGITransport(app=app), base_url=BASE_URL) as c:
        for task_id in known:
            r = await c.get(f"/tasks/{task_id}")
            assert r.status_code == 200
            row = r.json()
            assert row["task_id"] == task_id
            assert row["max_days"] > 0
            assert row["officer_pool_total"] > 0
            assert isinstance(row["services"], list)
            assert len(row["services"]) >= 1

        bad = await c.get("/tasks/fake_task")
    assert bad.status_code == 404


async def test_session_data_flow_reset_masks_step_trace_snapshot_grade_cancel() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url=BASE_URL) as c:
        reset = await c.post("/reset", json={"task_id": "district_backlog_easy", "seed": 42})
        assert reset.status_code == 200
        reset_body = reset.json()
        sid = reset_body["session_id"]
        assert len(sid) == 36
        assert reset_body["task_id"] == "district_backlog_easy"
        assert reset_body["observation"]["day"] == 0

        masks = await c.post("/action-masks", json={"session_id": sid})
        assert masks.status_code == 200
        mask_body = masks.json()
        assert len(mask_body["action_mask"]) == N_ACTIONS
        assert mask_body["total_actions"] == N_ACTIONS
        assert mask_body["total_valid"] > 0

        for _ in range(3):
            step = await c.post(
                "/step",
                json={"session_id": sid, "action": {"action_type": "advance_time"}},
            )
            assert step.status_code == 200

        state = await c.get("/state", params={"session_id": sid, "include_action_history": True})
        assert state.status_code == 200
        st = state.json()["state"]
        assert st["day"] >= 1
        assert st["action_history_count"] >= 3

        trace_page1 = await c.get(f"/simulate/{sid}/trace", params={"page": 1, "page_size": 2})
        trace_page2 = await c.get(f"/simulate/{sid}/trace", params={"page": 2, "page_size": 2})
        assert trace_page1.status_code == 200
        assert trace_page2.status_code == 200
        p1 = trace_page1.json()
        p2 = trace_page2.json()
        assert p1["total_steps"] >= 3
        assert len(p1["steps"]) == 2
        assert p2["page"] == 2
        assert len(p2["steps"]) >= 1

        snap = await c.get(f"/simulate/{sid}/snapshot")
        assert snap.status_code == 200
        snap_body = snap.json()
        assert snap_body["session_id"] == sid
        assert "observation" in snap_body

        grade = await c.post("/grade", json={"session_id": sid})
        assert grade.status_code == 200
        g = grade.json()
        assert g["task_id"] == "district_backlog_easy"
        assert 0.0 <= g["score"] <= 1.0
        assert isinstance(g["metrics"], dict)

        cancel = await c.post(f"/simulate/{sid}/cancel")
        assert cancel.status_code == 200
        assert cancel.json()["status"] == "cancelled"

        state_after = await c.get("/state", params={"session_id": sid})
        assert state_after.status_code == 404


async def test_simulate_endpoint_validation_contract() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url=BASE_URL, timeout=30.0) as c:
        bad_task = await c.post(
            "/simulate",
            json={
                "task_id": "not_a_real_task",
                "agent_mode": "baseline_policy",
                "max_steps": 3,
                "seed": 123,
            },
        )
        bad_mode = await c.post(
            "/simulate",
            json={
                "task_id": "district_backlog_easy",
                "agent_mode": "wrong_mode",
                "max_steps": 3,
                "seed": 123,
            },
        )

    assert bad_task.status_code == 422
    assert bad_mode.status_code == 422


async def test_rl_models_and_rl_run_missing_model_guardrail() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url=BASE_URL) as c:
        models = await c.get("/rl/models")
        assert models.status_code == 200
        rows = models.json()
        assert isinstance(rows, list)
        assert len(rows) >= 1
        for row in rows:
            assert "model_path" in row
            assert "exists" in row

        missing = await c.post(
            "/rl/run",
            json={
                "task_id": "district_backlog_easy",
                "model_path": "results/best_model/does_not_exist",
                "seed": 42,
                "max_steps": 10,
                "n_episodes": 1,
            },
        )
    assert missing.status_code == 422
