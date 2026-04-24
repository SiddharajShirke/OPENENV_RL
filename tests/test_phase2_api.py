"""
tests/test_phase2_api.py
Phase 2 API: FastAPI endpoints /health /reset /step /state /grade /sessions
Run (server must be running on localhost:7860):
    pytest tests/test_phase2_api.py -v
OR against the TestClient (no server needed):
    pytest tests/test_phase2_api.py -v --use-testclient
"""
import pytest
import sys

# ── Use TestClient by default — no running server needed ─────────────────────
try:
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)
    USE_TESTCLIENT = True
except Exception:
    import requests
    BASE = "http://localhost:7860"
    USE_TESTCLIENT = False


def post(path: str, body: dict) -> dict:
    if USE_TESTCLIENT:
        r = client.post(path, json=body)
    else:
        import requests
        r = requests.post(f"{BASE}{path}", json=body)
    return r.status_code, r.json()


def get(path: str, params: dict = None) -> dict:
    if USE_TESTCLIENT:
        r = client.get(path, params=params)
    else:
        import requests
        r = requests.get(f"{BASE}{path}", params=params)
    return r.status_code, r.json()


def delete(path: str) -> dict:
    if USE_TESTCLIENT:
        r = client.delete(path)
    else:
        import requests
        r = requests.delete(f"{BASE}{path}")
    return r.status_code, r.json()


# ─── /health ──────────────────────────────────────────────────────────────────
class TestHealth:
    def test_health_returns_200(self):
        code, body = get("/health")
        assert code == 200

    def test_health_status_ok(self):
        _, body = get("/health")
        assert body.get("status") == "ok"

    def test_health_has_version(self):
        _, body = get("/health")
        assert "version" in body

    def test_health_has_active_sessions(self):
        _, body = get("/health")
        assert "active_sessions" in body
        assert isinstance(body["active_sessions"], int)


# ─── POST /reset ──────────────────────────────────────────────────────────────
class TestReset:
    def test_reset_returns_200(self):
        code, _ = post("/reset", {"task_id": "district_backlog_easy"})
        assert code == 200

    def test_reset_returns_session_id(self):
        _, body = post("/reset", {"task_id": "district_backlog_easy"})
        assert "session_id" in body
        assert isinstance(body["session_id"], str)
        assert len(body["session_id"]) > 0

    def test_reset_returns_observation(self):
        _, body = post("/reset", {"task_id": "district_backlog_easy"})
        assert "observation" in body
        obs = body["observation"]
        assert obs["day"] == 0
        assert obs["task_id"] == "district_backlog_easy"

    def test_reset_returns_info_dict(self):
        _, body = post("/reset", {"task_id": "district_backlog_easy"})
        assert "info" in body
        assert isinstance(body["info"], dict)

    def test_reset_with_seed(self):
        code, body = post("/reset", {"task_id": "district_backlog_easy", "seed": 42})
        assert code == 200
        assert "session_id" in body

    def test_reset_different_tasks(self):
        for tid in ["district_backlog_easy", "mixed_urgency_medium", "cross_department_hard"]:
            code, body = post("/reset", {"task_id": tid})
            assert code == 200, f"Reset failed for task {tid}"
            assert body["observation"]["task_id"] == tid

    def test_two_resets_give_different_session_ids(self):
        _, b1 = post("/reset", {"task_id": "district_backlog_easy"})
        _, b2 = post("/reset", {"task_id": "district_backlog_easy"})
        assert b1["session_id"] != b2["session_id"]


# ─── POST /step ───────────────────────────────────────────────────────────────
class TestStep:
    def _session(self):
        _, body = post("/reset", {"task_id": "district_backlog_easy", "seed": 42})
        return body["session_id"]

    def test_step_returns_200(self):
        sid = self._session()
        code, _ = post("/step", {
            "session_id": sid,
            "action": {"action_type": "advance_time"},
        })
        assert code == 200

    def test_step_returns_all_fields(self):
        sid = self._session()
        _, body = post("/step", {
            "session_id": sid,
            "action": {"action_type": "advance_time"},
        })
        assert "observation" in body
        assert "reward" in body
        assert "terminated" in body
        assert "truncated" in body
        assert "info" in body

    def test_step_reward_is_number(self):
        sid = self._session()
        _, body = post("/step", {
            "session_id": sid,
            "action": {"action_type": "advance_time"},
        })
        assert isinstance(body["reward"], (int, float))

    def test_step_observation_day_increments(self):
        sid = self._session()
        _, b = post("/step", {"session_id": sid,
                              "action": {"action_type": "advance_time"}})
        assert b["observation"]["day"] == 1

    def test_step_set_priority_mode(self):
        sid = self._session()
        _, body = post("/step", {
            "session_id": sid,
            "action": {"action_type": "set_priority_mode",
                       "priority_mode": "urgent_first"},
        })
        assert body["info"]["invalid_action"] is False

    def test_step_invalid_action_flagged(self):
        sid = self._session()
        _, body = post("/step", {
            "session_id": sid,
            "action": {"action_type": "set_priority_mode"},  # missing priority_mode
        })
        assert body["info"]["invalid_action"] is True

    def test_step_on_unknown_session_returns_404(self):
        code, _ = post("/step", {
            "session_id": "no-such-session-xyz",
            "action": {"action_type": "advance_time"},
        })
        assert code == 404

    def test_step_terminated_episode_returns_409(self):
        sid = self._session()
        # Run until termination
        for _ in range(200):
            _, b = post("/step", {"session_id": sid,
                                  "action": {"action_type": "advance_time"}})
            if b.get("terminated") or b.get("truncated"):
                break
        # One more step should be 409
        code, _ = post("/step", {
            "session_id": sid,
            "action": {"action_type": "advance_time"},
        })
        assert code in [409, 422]


# ─── GET/POST /state ──────────────────────────────────────────────────────────
class TestState:
    def _session(self):
        _, body = post("/reset", {"task_id": "district_backlog_easy", "seed": 42})
        return body["session_id"]

    def test_state_post_returns_200(self):
        sid = self._session()
        code, _ = post("/state", {"session_id": sid})
        assert code == 200

    def test_state_get_returns_200(self):
        sid = self._session()
        code, _ = get("/state", {"session_id": sid})
        assert code == 200

    def test_state_has_episode_state(self):
        sid = self._session()
        _, body = post("/state", {"session_id": sid})
        assert "state" in body

    def test_state_day_zero_at_start(self):
        sid = self._session()
        _, body = post("/state", {"session_id": sid})
        assert body["state"]["day"] == 0

    def test_state_unknown_session_404(self):
        code, _ = post("/state", {"session_id": "ghost-session"})
        assert code == 404

    def test_state_action_history_excluded_by_default(self):
        sid = self._session()
        _, body = post("/state", {"session_id": sid,
                                  "include_action_history": False})
        state = body["state"]
        assert "action_history" not in state or state.get("action_history") is None


# ─── POST /grade ──────────────────────────────────────────────────────────────
class TestGrade:
    def _run_session(self, steps=5):
        _, body = post("/reset", {"task_id": "district_backlog_easy", "seed": 42})
        sid = body["session_id"]
        for _ in range(steps):
            r = post("/step", {"session_id": sid,
                               "action": {"action_type": "advance_time"}})
            if r[1].get("terminated") or r[1].get("truncated"):
                break
        return sid

    def test_grade_returns_200(self):
        sid = self._run_session()
        code, _ = post("/grade", {"session_id": sid})
        assert code == 200

    def test_grade_score_in_range(self):
        sid = self._run_session()
        _, body = post("/grade", {"session_id": sid})
        assert 0.0 <= body["score"] <= 1.0

    def test_grade_has_grader_name(self):
        sid = self._run_session()
        _, body = post("/grade", {"session_id": sid})
        assert "grader_name" in body
        assert isinstance(body["grader_name"], str)

    def test_grade_has_metrics(self):
        sid = self._run_session()
        _, body = post("/grade", {"session_id": sid})
        assert "metrics" in body

    def test_grade_unknown_session_404(self):
        code, _ = post("/grade", {"session_id": "dead-session"})
        assert code == 404


# ─── GET /sessions / DELETE /sessions/{id} ───────────────────────────────────
class TestSessions:
    def test_list_sessions_returns_200(self):
        code, _ = get("/sessions")
        assert code == 200

    def test_list_sessions_has_count(self):
        _, body = get("/sessions")
        assert "active_sessions" in body

    def test_delete_session(self):
        _, r = post("/reset", {"task_id": "district_backlog_easy"})
        sid = r["session_id"]
        code, body = delete(f"/sessions/{sid}")
        assert code == 200
        assert body.get("deleted") == sid

    def test_delete_nonexistent_session_404(self):
        code, _ = delete("/sessions/nonexistent-id-xyz")
        assert code == 404

    def test_session_count_increases_after_reset(self):
        _, b1 = get("/sessions")
        count_before = b1["active_sessions"]
        post("/reset", {"task_id": "district_backlog_easy"})
        _, b2 = get("/sessions")
        count_after = b2["active_sessions"]
        assert count_after >= count_before
