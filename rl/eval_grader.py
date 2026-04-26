"""
Grader-based evaluation utility for trained RL checkpoints.

This complements `rl/evaluate.py`:
- `rl/evaluate.py` is batch-oriented and returns aggregate task rows.
- `rl/eval_grader.py` is phase/task-oriented and prints per-episode progress,
  promotion guidance, and an optional score/reward plot.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Literal

import matplotlib
import numpy as np
from sb3_contrib import MaskablePPO, RecurrentPPO

# Allow running as `python rl/eval_grader.py ...` from repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.graders import grade_episode
from rl.gov_workflow_env import GovWorkflowGymEnv

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ModelType = Literal["auto", "maskable", "recurrent"]

PROMOTION_THRESHOLDS = {
    "district_backlog_easy": 0.75,
    "mixed_urgency_medium": 0.65,
    "cross_department_hard": 0.55,
}

PHASE_LABELS = {
    "district_backlog_easy": "Phase 1",
    "mixed_urgency_medium": "Phase 2",
    "cross_department_hard": "Phase 3",
}


def _normalize_action(action: Any) -> int:
    if isinstance(action, np.ndarray):
        return int(action.item())
    return int(action)


def _sanitize_action(action_idx: int, masks: np.ndarray) -> int:
    if 0 <= action_idx < masks.shape[0] and bool(masks[action_idx]):
        return int(action_idx)
    if masks.shape[0] > 18 and bool(masks[18]):
        return 18
    valid = np.flatnonzero(masks)
    return int(valid[0]) if valid.size > 0 else 18


def _load_model(model_path: str, model_type: ModelType) -> tuple[Any, str]:
    if model_type == "maskable":
        return MaskablePPO.load(model_path), "maskable"
    if model_type == "recurrent":
        return RecurrentPPO.load(model_path), "recurrent"

    try:
        return MaskablePPO.load(model_path), "maskable"
    except Exception:
        return RecurrentPPO.load(model_path), "recurrent"


def evaluate_with_grader(
    model_path: str,
    task_id: str,
    n_episodes: int = 20,
    seed: int = 42,
    model_type: ModelType = "auto",
    save_plot: bool = True,
) -> float:
    if task_id not in PROMOTION_THRESHOLDS:
        allowed = ", ".join(PROMOTION_THRESHOLDS.keys())
        raise ValueError(f"Unknown task_id '{task_id}'. Allowed: {allowed}")

    model, resolved_type = _load_model(model_path, model_type)

    print("\n" + "=" * 64)
    print(f"Track A Evaluation - {PHASE_LABELS.get(task_id, task_id)}")
    print(f"Model: {model_path}")
    print(f"Model type: {resolved_type}")
    print(f"Task: {task_id}")
    print(f"Episodes: {n_episodes}")
    print("=" * 64 + "\n")

    scores: list[float] = []
    rewards: list[float] = []

    for ep in range(n_episodes):
        env = GovWorkflowGymEnv(task_id=task_id, seed=seed + ep, hard_action_mask=True)
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_reward = 0.0
        lstm_state: Any = None
        episode_start = np.array([True], dtype=bool)

        while not done:
            masks = env.action_masks()
            if resolved_type == "recurrent":
                action, lstm_state = model.predict(
                    obs,
                    state=lstm_state,
                    episode_start=episode_start,
                    deterministic=True,
                )
                action_idx = _sanitize_action(_normalize_action(action), masks)
            else:
                action, _ = model.predict(obs, action_masks=masks, deterministic=True)
                action_idx = _normalize_action(action)

            obs, reward, terminated, truncated, _ = env.step(action_idx)
            ep_reward += float(reward)
            done = bool(terminated or truncated)
            episode_start = np.array([done], dtype=bool)

        result = grade_episode(env.core_env.state())
        score = float(result.score)
        threshold = float(PROMOTION_THRESHOLDS[task_id])
        badge = "PASS" if score >= threshold else "FAIL"
        print(f"  {badge:4} ep={ep + 1:02d} score={score:.4f} reward={ep_reward:.2f}")
        scores.append(score)
        rewards.append(ep_reward)

    mean_score = float(np.mean(scores)) if scores else 0.0
    threshold = float(PROMOTION_THRESHOLDS[task_id])

    print("\n" + "-" * 64)
    print(f"Mean grader score: {mean_score:.4f}")
    print(f"Promotion target : {threshold:.2f}")
    print(f"Min / Max        : {float(np.min(scores)):.4f} / {float(np.max(scores)):.4f}")
    print(f"Pass rate        : {sum(s >= threshold for s in scores)}/{len(scores)}")
    if mean_score >= threshold:
        print("Decision         : PROMOTE")
    else:
        print("Decision         : CONTINUE TRAINING")
    print("=" * 64)

    if save_plot:
        _save_plot(scores=scores, rewards=rewards, task_id=task_id, mean_score=mean_score, threshold=threshold, model_path=model_path)

    return mean_score


def _save_plot(
    *,
    scores: list[float],
    rewards: list[float],
    task_id: str,
    mean_score: float,
    threshold: float,
    model_path: str,
) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Track A - {PHASE_LABELS.get(task_id, task_id)} Evaluation\n"
        f"Task: {task_id} | Model: {os.path.basename(model_path)}",
        fontsize=12,
        fontweight="bold",
    )

    episodes = list(range(1, len(scores) + 1))

    ax1 = axes[0]
    colors = ["#0e8a16" if s >= threshold else "#b60205" for s in scores]
    ax1.bar(episodes, scores, color=colors, alpha=0.85)
    ax1.axhline(y=threshold, color="#d97706", linestyle="--", linewidth=2, label=f"threshold={threshold:.2f}")
    ax1.axhline(y=mean_score, color="#1d4ed8", linestyle="-", linewidth=2, label=f"mean={mean_score:.3f}")
    ax1.set_ylim(0.0, 1.05)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Grader Score")
    ax1.set_title("Per-Episode Grader Score")
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.legend()

    ax2 = axes[1]
    ax2.plot(episodes, rewards, color="#0369a1", linewidth=2, marker="o", markersize=4)
    if rewards:
        mean_reward = float(np.mean(rewards))
        ax2.axhline(y=mean_reward, color="#d97706", linestyle="--", linewidth=2, label=f"mean={mean_reward:.2f}")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Total Reward")
    ax2.set_title("Episode Reward")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    out_dir = os.path.join("results", "eval_logs", task_id)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{task_id}_grader_eval.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Plot saved -> {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Task-oriented grader evaluation for a trained checkpoint")
    parser.add_argument("--model", required=True, help="Path to .zip checkpoint (suffix optional)")
    parser.add_argument("--task", required=True, choices=list(PROMOTION_THRESHOLDS.keys()))
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-type", choices=["auto", "maskable", "recurrent"], default="auto")
    parser.add_argument("--no-plot", action="store_true", help="Disable PNG output")
    args = parser.parse_args()

    model_path = args.model if args.model.endswith(".zip") else f"{args.model}.zip"
    evaluate_with_grader(
        model_path=model_path,
        task_id=args.task,
        n_episodes=args.episodes,
        seed=args.seed,
        model_type=args.model_type,
        save_plot=not args.no_plot,
    )


if __name__ == "__main__":
    main()
