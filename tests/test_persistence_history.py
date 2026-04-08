from fastapi.testclient import TestClient

from app.main import app


def test_simulation_history_persists_completed_runs() -> None:
    client = TestClient(app)

    run_resp = client.post(
        "/api/simulation/run",
        json={
            "task_id": "district_backlog_easy",
            "agent_mode": "baseline_policy",
            "policy_name": "backlog_clearance",
            "max_steps": 5,
            "seed": 123,
        },
    )
    assert run_resp.status_code == 200

    history_resp = client.get("/api/history/simulations")
    assert history_resp.status_code == 200
    runs = history_resp.json().get("runs", [])
    assert isinstance(runs, list)
    assert any(row.get("task_id") == "district_backlog_easy" for row in runs)

    run_id = next((row.get("run_id") for row in runs if row.get("run_id")), None)
    assert run_id
    detail_resp = client.get(f"/api/history/simulations/{run_id}")
    assert detail_resp.status_code == 200
    detail = detail_resp.json()
    assert detail.get("run_id") == run_id


def test_comparison_history_roundtrip() -> None:
    client = TestClient(app)

    payload = {
        "task_id": "district_backlog_easy",
        "baseline_policy": "backlog_clearance",
        "model_path": "results/best_model/phase2_final.zip",
        "model_type": "maskable",
        "include_llm": True,
        "runs": 2,
        "steps": 10,
        "episodes": 1,
        "seed_base": 100,
        "result": {
            "baselineScore": 0.6,
            "trainedScore": 0.7,
            "llmScore": 0.5,
        },
    }
    create_resp = client.post("/api/history/comparisons", json=payload)
    assert create_resp.status_code == 200
    comparison_id = create_resp.json().get("comparison_id")
    assert comparison_id

    list_resp = client.get("/api/history/comparisons")
    assert list_resp.status_code == 200
    rows = list_resp.json().get("comparisons", [])
    assert any(row.get("comparison_id") == comparison_id for row in rows)

    detail_resp = client.get(f"/api/history/comparisons/{comparison_id}")
    assert detail_resp.status_code == 200
    assert detail_resp.json().get("comparison_id") == comparison_id
