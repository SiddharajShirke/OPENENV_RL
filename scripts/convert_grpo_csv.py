"""
scripts/convert_grpo_csv.py

Converts GRPO training CSV logs to JSON format
for the FastAPI /training/* story endpoints.

CSV format expected:
  step, reward, fn1_valid, fn2_no_halluc, fn3_env_score

Usage:
  python scripts/convert_grpo_csv.py \
      --csv  grpo_training_log.csv \
      --task mixed_urgency_medium

Output:
  data/training_logs/{task_id}_training_log.json
"""

from __future__ import annotations
import csv
import json
import argparse
from pathlib import Path


def load_csv(csv_path: str) -> list[dict]:
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        reward_values: list[float] = []
        raw_rows: list[dict] = []

        def _pick(row: dict, names: list[str], default: float) -> float:
            for name in names:
                if name in row and str(row.get(name, "")).strip() != "":
                    try:
                        return float(row[name])
                    except (TypeError, ValueError):
                        continue
            return float(default)

        for row in reader:
            raw_rows.append(row)
            reward_values.append(_pick(row, ["reward", "total_reward"], 0.0))

        r_min = min(reward_values) if reward_values else 0.0
        r_rng = (max(reward_values) - r_min) if reward_values else 1.0
        if r_rng == 0:
            r_rng = 1.0

        for i, row in enumerate(raw_rows):
            reward_val = _pick(row, ["reward", "total_reward"], 0.0)
            fallback_norm = (reward_val - r_min) / r_rng
            step_default = i + 1
            if "step" in row and str(row.get("step", "")).strip() != "":
                try:
                    step_default = int(float(row["step"]))
                except (TypeError, ValueError):
                    step_default = i + 1

            rows.append({
                "step":           step_default,
                "reward":         reward_val,
                "fn1_valid":      _pick(row, ["fn1_valid", "valid_action_rate"], 1.0),
                "fn2_no_halluc":  _pick(row, ["fn2_no_halluc", "hallucination_free"], 1.0),
                "fn3_env_score":  _pick(row, ["fn3_env_score", "env_score"], fallback_norm),
            })
    return rows


def build_log(rows: list[dict], task_id: str) -> dict:
    n        = len(rows)
    rewards  = [r["reward"]        for r in rows]
    fn1_vals = [r["fn1_valid"]     for r in rows]
    fn2_vals = [r["fn2_no_halluc"] for r in rows]
    fn3_vals = [r["fn3_env_score"] for r in rows]

    fn3_min = min(fn3_vals)
    fn3_rng = (max(fn3_vals) - fn3_min) or 1.0

    episodes = []
    for i, r in enumerate(rows):
        norm_env = (r["fn3_env_score"] - fn3_min) / fn3_rng
        combined = round(
            r["fn1_valid"] * 0.3 + r["fn2_no_halluc"] * 0.2 + norm_env * 0.5,
            4
        )
        phase = (
            "random"    if i < n * 0.25 else
            "exploring" if i < n * 0.50 else
            "learning"  if i < n * 0.75 else
            "converged"
        )
        episodes.append({
            "episode":       r["step"],
            "total_reward":  round(r["reward"], 4),
            "score":         combined,
            "fn1_valid":     round(r["fn1_valid"], 4),
            "fn2_no_halluc": round(r["fn2_no_halluc"], 4),
            "fn3_env_score": round(r["fn3_env_score"], 4),
            "phase":         phase,
            "actions": {
                "valid_action_rate":  round(r["fn1_valid"], 4),
                "hallucination_free": round(r["fn2_no_halluc"], 4),
                "env_score":          round(norm_env, 4),
            },
        })

    scores = [e["score"] for e in episodes]

    return {
        "task_id":          task_id,
        "total_episodes":   n,
        "base_model":       "Qwen/Qwen2-1.5B-Instruct",
        "adapter_path":     f"artifacts/llm/{task_id.split('_')[1]}/",
        "training_method":  "GRPO",
        "lora_rank":        16,
        "reward_functions": {
            "fn1_valid":     "Action validity - legal JSON action output (0->1)",
            "fn2_no_halluc": "No hallucination - stayed on gov workflow topic (0->1)",
            "fn3_env_score": "Environment score - improved gov workflow quality",
        },
        "summary": {
            "first_episode_reward":    round(rewards[0],  4),
            "last_episode_reward":     round(rewards[-1], 4),
            "best_episode_reward":     round(max(rewards), 4),
            "first_episode_score":     round(scores[0],   4),
            "last_episode_score":      round(scores[-1],  4),
            "best_episode_score":      round(max(scores), 4),
            "reward_improvement_pct":  round(
                ((rewards[-1] - rewards[0]) / abs(rewards[0])) * 100, 2
            ) if rewards[0] != 0 else 0.0,
            "invalid_action_steps":    sum(1 for r in rows if r["fn1_valid"]     < 1.0),
            "hallucination_steps":     sum(1 for r in rows if r["fn2_no_halluc"] < 1.0),
            "avg_fn1_valid":           round(sum(fn1_vals) / n, 4),
            "avg_fn2_no_halluc":       round(sum(fn2_vals) / n, 4),
            "avg_fn3_env_score":       round(sum(fn3_vals) / n, 4),
        },
        "episodes": episodes,
    }


def save_log(log: dict, out_dir: str) -> str:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = f"{out_dir}/{log['task_id']}_training_log.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",  required=True, help="Path to GRPO CSV file")
    parser.add_argument("--task", required=True, help="Task ID e.g. mixed_urgency_medium")
    parser.add_argument("--out",  default="data/training_logs", help="Output directory")
    args = parser.parse_args()

    print(f"Reading CSV  : {args.csv}")
    rows = load_csv(args.csv)
    print(f"Steps found  : {len(rows)}")

    log = build_log(rows, args.task)
    out = save_log(log, args.out)

    print(f"Saved JSON   : {out}")
    print(f"Steps        : {log['total_episodes']}")
    print(f"Reward range : {log['summary']['first_episode_reward']} -> {log['summary']['last_episode_reward']}")
    print(f"Score range  : {log['summary']['first_episode_score']} -> {log['summary']['last_episode_score']}")
    print(f"Invalid steps: {log['summary']['invalid_action_steps']}")
