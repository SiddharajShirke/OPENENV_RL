"""
Live HTTP audit for Gov Workflow OpenEnv API.

This script calls the full 16-endpoint contract over real HTTP
and writes a timestamped JSON report with pass/fail + response samples.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, request


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _shorten(text: str, max_chars: int = 800) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "...<truncated>"


def _http_call(
    base_url: str,
    method: str,
    path: str,
    *,
    body: dict[str, Any] | None = None,
    timeout_sec: int = 30,
    max_sample_chars: int = 800,
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}{path}"
    payload_bytes = None
    headers = {"Accept": "application/json"}
    if body is not None:
        payload_bytes = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = request.Request(
        url=url,
        data=payload_bytes,
        headers=headers,
        method=method.upper(),
    )

    status_code = None
    raw_text = ""
    parsed_json = None
    err_text = None

    try:
        with request.urlopen(req, timeout=timeout_sec) as resp:
            status_code = int(resp.status)
            raw_text = resp.read().decode("utf-8", errors="replace")
    except error.HTTPError as exc:
        status_code = int(exc.code)
        raw_text = exc.read().decode("utf-8", errors="replace")
        err_text = str(exc)
    except Exception as exc:  # network/timeout etc.
        err_text = str(exc)

    if raw_text:
        try:
            parsed_json = json.loads(raw_text)
        except Exception:
            parsed_json = None

    return {
        "method": method.upper(),
        "path": path,
        "url": url,
        "request_body": body,
        "status_code": status_code,
        "ok": err_text is None or status_code is not None,
        "error": err_text,
        "response_json": parsed_json,
        "response_text_sample": _shorten(raw_text, max_chars=max_sample_chars),
    }


def _extract_sse_data_lines(raw_text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in raw_text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:") :].strip()
        if not payload:
            continue
        try:
            rows.append(json.loads(payload))
        except Exception:
            rows.append({"raw": payload})
    return rows


def run_audit(base_url: str, timeout_sec: int = 30) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    context: dict[str, Any] = {}

    def add_check(name: str, call_result: dict[str, Any], expected_statuses: list[int]) -> None:
        status_code = call_result.get("status_code")
        passed = bool(status_code in expected_statuses)
        checks.append(
            {
                "name": name,
                "endpoint": f"{call_result['method']} {call_result['path']}",
                "expected_statuses": expected_statuses,
                "status_code": status_code,
                "passed": passed,
                "error": call_result.get("error"),
                "response_sample": call_result.get("response_json")
                if call_result.get("response_json") is not None
                else call_result.get("response_text_sample"),
            }
        )
        call_result["passed"] = passed

    # 1) /health
    r = _http_call(base_url, "GET", "/health", timeout_sec=timeout_sec)
    add_check("health", r, [200])

    # 2) /tasks
    r = _http_call(base_url, "GET", "/tasks", timeout_sec=timeout_sec)
    add_check("tasks", r, [200])
    task_ids: list[str] = []
    if isinstance(r.get("response_json"), list):
        task_ids = [str(x.get("task_id")) for x in r["response_json"] if isinstance(x, dict) and x.get("task_id")]
    if not task_ids:
        task_ids = ["district_backlog_easy", "mixed_urgency_medium", "cross_department_hard"]
    context["task_ids"] = task_ids

    # 3) /tasks/{task_id} (test first available)
    task_id = task_ids[0]
    r = _http_call(base_url, "GET", f"/tasks/{task_id}", timeout_sec=timeout_sec)
    add_check("task_detail", r, [200])

    # 4) /metrics
    r = _http_call(base_url, "GET", "/metrics", timeout_sec=timeout_sec)
    add_check("metrics", r, [200])

    # 5) /actions/schema
    r = _http_call(base_url, "GET", "/actions/schema", timeout_sec=timeout_sec)
    add_check("actions_schema", r, [200])

    # 6) /rl/models
    r = _http_call(base_url, "GET", "/rl/models", timeout_sec=timeout_sec)
    add_check("rl_models", r, [200])

    # 7) /reset
    r = _http_call(
        base_url,
        "POST",
        "/reset",
        body={"task_id": task_id, "seed": 42},
        timeout_sec=timeout_sec,
    )
    add_check("reset", r, [200])
    sid = None
    if isinstance(r.get("response_json"), dict):
        sid = r["response_json"].get("session_id")
    context["session_id"] = sid

    # 8) /action-masks
    if sid:
        r = _http_call(
            base_url,
            "POST",
            "/action-masks",
            body={"session_id": sid},
            timeout_sec=timeout_sec,
        )
        add_check("action_masks", r, [200])
    else:
        checks.append(
            {
                "name": "action_masks",
                "endpoint": "POST /action-masks",
                "expected_statuses": [200],
                "status_code": None,
                "passed": False,
                "error": "Skipped: no session_id from /reset",
                "response_sample": None,
            }
        )

    # 9) /step
    if sid:
        r = _http_call(
            base_url,
            "POST",
            "/step",
            body={"session_id": sid, "action": {"action_type": "advance_time"}},
            timeout_sec=timeout_sec,
        )
        add_check("step", r, [200])
    else:
        checks.append(
            {
                "name": "step",
                "endpoint": "POST /step",
                "expected_statuses": [200],
                "status_code": None,
                "passed": False,
                "error": "Skipped: no session_id from /reset",
                "response_sample": None,
            }
        )

    # 10) /state
    if sid:
        r = _http_call(
            base_url,
            "GET",
            f"/state?session_id={sid}&include_action_history=true",
            timeout_sec=timeout_sec,
        )
        add_check("state", r, [200])
    else:
        checks.append(
            {
                "name": "state",
                "endpoint": "GET /state",
                "expected_statuses": [200],
                "status_code": None,
                "passed": False,
                "error": "Skipped: no session_id from /reset",
                "response_sample": None,
            }
        )

    # 11) /simulate (SSE)
    r = _http_call(
        base_url,
        "POST",
        "/simulate",
        body={"task_id": task_id, "agent_mode": "baseline_policy", "max_steps": 3, "seed": 42},
        timeout_sec=timeout_sec,
        max_sample_chars=4000,
    )
    parsed_rows = _extract_sse_data_lines(r.get("response_text_sample", ""))
    has_step = any(isinstance(x, dict) and "step" in x for x in parsed_rows)
    has_done = any(isinstance(x, dict) and x.get("done") is True for x in parsed_rows)
    simulate_pass = (r.get("status_code") == 200) and has_step and has_done
    checks.append(
        {
            "name": "simulate_stream",
            "endpoint": "POST /simulate",
            "expected_statuses": [200],
            "status_code": r.get("status_code"),
            "passed": simulate_pass,
            "error": r.get("error"),
            "response_sample": {
                "sse_rows_sample": parsed_rows[:3],
                "has_step": has_step,
                "has_done": has_done,
            },
        }
    )

    # 12) /simulate/{session_id}/snapshot
    if sid:
        r = _http_call(base_url, "GET", f"/simulate/{sid}/snapshot", timeout_sec=timeout_sec)
        add_check("simulate_snapshot", r, [200])
    else:
        checks.append(
            {
                "name": "simulate_snapshot",
                "endpoint": "GET /simulate/{session_id}/snapshot",
                "expected_statuses": [200],
                "status_code": None,
                "passed": False,
                "error": "Skipped: no session_id from /reset",
                "response_sample": None,
            }
        )

    # 13) /simulate/{session_id}/trace
    if sid:
        r = _http_call(base_url, "GET", f"/simulate/{sid}/trace?page=1&page_size=20", timeout_sec=timeout_sec)
        add_check("simulate_trace", r, [200])
    else:
        checks.append(
            {
                "name": "simulate_trace",
                "endpoint": "GET /simulate/{session_id}/trace",
                "expected_statuses": [200],
                "status_code": None,
                "passed": False,
                "error": "Skipped: no session_id from /reset",
                "response_sample": None,
            }
        )

    # 14) /grade
    if sid:
        r = _http_call(base_url, "POST", "/grade", body={"session_id": sid}, timeout_sec=timeout_sec)
        add_check("grade", r, [200])
    else:
        checks.append(
            {
                "name": "grade",
                "endpoint": "POST /grade",
                "expected_statuses": [200],
                "status_code": None,
                "passed": False,
                "error": "Skipped: no session_id from /reset",
                "response_sample": None,
            }
        )

    # 15) /rl/run (guardrail: missing model -> 422)
    r = _http_call(
        base_url,
        "POST",
        "/rl/run",
        body={
            "task_id": task_id,
            "model_path": "results/best_model/does_not_exist",
            "seed": 42,
            "max_steps": 10,
            "n_episodes": 1,
        },
        timeout_sec=timeout_sec,
    )
    add_check("rl_run_missing_model_guardrail", r, [422])

    # 16) /simulate/{session_id}/cancel
    if sid:
        r = _http_call(base_url, "POST", f"/simulate/{sid}/cancel", timeout_sec=timeout_sec)
        add_check("simulate_cancel", r, [200])
    else:
        checks.append(
            {
                "name": "simulate_cancel",
                "endpoint": "POST /simulate/{session_id}/cancel",
                "expected_statuses": [200],
                "status_code": None,
                "passed": False,
                "error": "Skipped: no session_id from /reset",
                "response_sample": None,
            }
        )

    total = len(checks)
    passed = sum(1 for c in checks if c["passed"])
    failed = total - passed

    return {
        "audit_name": "gov-workflow-openenv-live-http-audit",
        "generated_at_utc": _now_iso(),
        "base_url": base_url,
        "summary": {
            "total_checks": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": round((passed / total) * 100.0, 2) if total else 0.0,
        },
        "context": context,
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:7860")
    parser.add_argument("--timeout-sec", type=int, default=30)
    parser.add_argument("--out-dir", default="reports/api_audit")
    args = parser.parse_args()

    report = run_audit(args.base_url, timeout_sec=args.timeout_sec)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"api_live_audit_{stamp}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Report written: {out_path}")
    print(
        f"Summary: passed={report['summary']['passed']}, "
        f"failed={report['summary']['failed']}, "
        f"pass_rate={report['summary']['pass_rate']}%"
    )
    if report["summary"]["failed"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
