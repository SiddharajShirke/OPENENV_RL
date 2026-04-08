#!/usr/bin/env python3
"""
Minimal smoke test for all benchmark tasks.

Runs one deterministic baseline episode per task and checks score bounds.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.baselines import run_policy_episode
from app.tasks import list_tasks


def main() -> int:
    results: list[dict] = []
    for task_id in list_tasks():
        result = run_policy_episode(task_id=task_id, policy_name="backlog_clearance")
        score = float(result["score"])
        if not (0.0 <= score <= 1.0):
            print(f"[FAIL] {task_id}: score out of range {score}")
            return 1
        results.append(result)
        print(
            f"[OK] task={task_id} score={score:.4f} "
            f"steps={result['steps']} completed={result['completed']}"
        )

    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "smoke_test_results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[DONE] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
