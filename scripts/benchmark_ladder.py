"""
Benchmark Ladder - compare all agents on all 3 tasks.

Usage:
    python scripts/benchmark_ladder.py
    python scripts/benchmark_ladder.py --phase1 results/best_model/phase1_final
"""

from __future__ import annotations

import argparse
import json
import os

from app.baselines import run_policy_episode
from rl.evaluate import TASK_IDS, evaluate_model


def fmt(v):
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def print_table(rows):
    print("\n" + "=" * 65)
    print(f"{'Agent':<28} {'Easy':>8} {'Medium':>8} {'Hard':>8} {'Avg':>8}")
    print("-" * 65)
    for r in rows:
        print(
            f"{r['agent']:<28} "
            f"{r.get('district_backlog_easy', '-'):>8} "
            f"{r.get('mixed_urgency_medium', '-'):>8} "
            f"{r.get('cross_department_hard', '-'):>8} "
            f"{r.get('average', '-'):>8}"
        )
    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1", default=None)
    parser.add_argument("--phase2", default=None)
    parser.add_argument("--phase3", default=None)
    parser.add_argument("--output", default="results/benchmark_ladder.json")
    args = parser.parse_args()

    all_rows = []

    for policy in ["urgent_first", "oldest_first", "backlog_clearance"]:
        row, scores = {"agent": f"heuristic_{policy}"}, []
        for tid in TASK_IDS:
            try:
                result = run_policy_episode(task_id=tid, policy_name=policy)
                s = float(result["score"])
                row[tid] = fmt(s)
                scores.append(s)
            except Exception:
                row[tid] = "ERR"
        row["average"] = fmt(sum(scores) / len(scores)) if scores else "-"
        all_rows.append(row)

    for label, path in [
        ("masked_ppo_ph1", args.phase1),
        ("curriculum_ppo_ph2", args.phase2),
        ("recurrent_ppo_ph3", args.phase3),
    ]:
        if not path or not os.path.exists(path + ".zip"):
            continue
        row, scores = {"agent": label}, []
        for r in evaluate_model(path, task_ids=TASK_IDS, verbose=False):
            row[r.task_id] = fmt(r.grader_score)
            scores.append(r.grader_score)
        row["average"] = fmt(sum(scores) / len(scores)) if scores else "-"
        all_rows.append(row)

    print_table(all_rows)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_rows, f, indent=2)
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
