"""
Generate training-curve evidence plots for Track A.

The script first tries monitor CSV files, then falls back to TensorBoard events.
It is read-only for training artifacts and does not trigger training.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

THRESHOLDS = {
    "district_backlog_easy": 0.75,
    "mixed_urgency_medium": 0.65,
    "cross_department_hard": 0.55,
}

PHASE_TO_RUN_DIR = {
    1: os.path.join("results", "runs", "phase1_masked_ppo"),
    2: os.path.join("results", "runs", "phase2_curriculum_ppo"),
    3: os.path.join("results", "runs", "phase3_recurrent_ppo"),
}


def _read_monitor_csv(monitor_path: str) -> tuple[list[float], list[float]]:
    rewards: list[float] = []
    lengths: list[float] = []
    with open(monitor_path, "r", encoding="utf-8") as f:
        # First line is metadata starting with '#'
        first = f.readline()
        if not first:
            return rewards, lengths
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rewards.append(float(row.get("r", 0.0)))
                lengths.append(float(row.get("l", 0.0)))
            except (TypeError, ValueError):
                continue
    return rewards, lengths


def _load_tb_scalars(event_path: str) -> dict[str, tuple[list[int], list[float]]]:
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except Exception:
        return {}

    try:
        acc = event_accumulator.EventAccumulator(event_path)
        acc.Reload()
        out: dict[str, tuple[list[int], list[float]]] = {}
        for tag in acc.Tags().get("scalars", []):
            vals = acc.Scalars(tag)
            out[tag] = ([int(v.step) for v in vals], [float(v.value) for v in vals])
        return out
    except Exception:
        return {}


def _latest_file(paths: list[str]) -> str | None:
    if not paths:
        return None
    return max(paths, key=lambda p: os.path.getmtime(p))


def _rolling(values: list[float], window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    w = max(1, int(window))
    kernel = np.ones(w, dtype=np.float64) / float(w)
    if arr.size < w:
        return np.full_like(arr, np.mean(arr))
    return np.convolve(arr, kernel, mode="same")


def plot_training(task_id: str, phase: int = 1) -> str:
    if task_id not in THRESHOLDS:
        allowed = ", ".join(THRESHOLDS.keys())
        raise ValueError(f"Unknown task_id '{task_id}'. Allowed: {allowed}")
    if phase not in PHASE_TO_RUN_DIR:
        raise ValueError("phase must be one of: 1, 2, 3")

    threshold = THRESHOLDS[task_id]
    run_dir = PHASE_TO_RUN_DIR[phase]

    monitor_candidates = glob.glob(os.path.join(run_dir, "**", "monitor.csv"), recursive=True)
    monitor_path = _latest_file(monitor_candidates)

    rewards: list[float] = []
    lengths: list[float] = []
    source = "none"

    if monitor_path and os.path.exists(monitor_path):
        rewards, lengths = _read_monitor_csv(monitor_path)
        source = f"monitor:{monitor_path}"
    else:
        event_candidates = glob.glob(os.path.join(run_dir, "**", "events.out.tfevents.*"), recursive=True)
        event_path = _latest_file(event_candidates)
        if event_path:
            scalars = _load_tb_scalars(event_path)
            rew_tag = "rollout/ep_rew_mean"
            len_tag = "rollout/ep_len_mean"
            if rew_tag in scalars:
                rewards = scalars[rew_tag][1]
            if len_tag in scalars:
                lengths = scalars[len_tag][1]
            source = f"tensorboard:{event_path}"

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        f"Track A - Phase {phase} Training Results\n"
        f"Task: {task_id} | Source: {source}",
        fontsize=13,
        fontweight="bold",
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.30)

    # Panel 1: reward trend
    ax1 = fig.add_subplot(gs[0, 0])
    if rewards:
        xs = np.arange(1, len(rewards) + 1)
        ax1.plot(xs, rewards, color="#0f766e", alpha=0.35, linewidth=1.2, label="raw")
        win = max(10, len(rewards) // 40)
        ax1.plot(xs, _rolling(rewards, win), color="#0f766e", linewidth=2.3, label=f"rolling(w={win})")
        ax1.set_title("Episode Reward Trend", fontweight="bold")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Reward")
        ax1.grid(True, alpha=0.3)
        ax1.legend()
    else:
        ax1.text(0.5, 0.5, "No reward data found", ha="center", va="center", transform=ax1.transAxes)
        ax1.set_title("Episode Reward Trend", fontweight="bold")

    # Panel 2: episode length trend
    ax2 = fig.add_subplot(gs[0, 1])
    if lengths:
        xs = np.arange(1, len(lengths) + 1)
        ax2.plot(xs, lengths, color="#7c3aed", alpha=0.35, linewidth=1.2, label="raw")
        win = max(10, len(lengths) // 40)
        ax2.plot(xs, _rolling(lengths, win), color="#7c3aed", linewidth=2.3, label=f"rolling(w={win})")
        ax2.set_title("Episode Length Trend", fontweight="bold")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Length")
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, "No length data found", ha="center", va="center", transform=ax2.transAxes)
        ax2.set_title("Episode Length Trend", fontweight="bold")

    # Panel 3: final-quarter reward distribution
    ax3 = fig.add_subplot(gs[1, 0])
    if rewards:
        start_idx = (len(rewards) * 3) // 4
        final_chunk = rewards[start_idx:] or rewards
        ax3.hist(final_chunk, bins=20, color="#15803d", alpha=0.82, edgecolor="white")
        ax3.axvline(float(np.mean(final_chunk)), color="#d97706", linewidth=2, label=f"mean={np.mean(final_chunk):.2f}")
        ax3.set_title("Final-Quarter Reward Distribution", fontweight="bold")
        ax3.set_xlabel("Reward")
        ax3.set_ylabel("Frequency")
        ax3.grid(True, alpha=0.3, axis="y")
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, "No reward distribution available", ha="center", va="center", transform=ax3.transAxes)
        ax3.set_title("Final-Quarter Reward Distribution", fontweight="bold")

    # Panel 4: configuration summary
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")

    metadata = {}
    meta_path = os.path.join("results", "best_model", f"phase{phase}_metadata.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception:
            metadata = {}

    summary = (
        f"Phase {phase} Summary\n"
        f"{'-' * 36}\n"
        f"Task:             {task_id}\n"
        f"Promotion target: >= {threshold:.2f}\n"
        f"Run directory:    {run_dir}\n"
        f"Data source:      {source}\n"
        f"Reward points:    {len(rewards)}\n"
        f"Length points:    {len(lengths)}\n"
        f"Algorithm:        {metadata.get('algorithm', 'PPO family')}\n"
        f"Architecture:     {metadata.get('architecture', 'MLP / LSTM as configured')}\n"
        f"Timesteps:        {metadata.get('timesteps', 'n/a')}\n"
        f"n_envs:           {metadata.get('n_envs', 'n/a')}\n"
        f"Seed:             {metadata.get('seed', 'n/a')}\n"
    )
    ax4.text(
        0.03,
        0.97,
        summary,
        transform=ax4.transAxes,
        verticalalignment="top",
        fontsize=9.5,
        family="monospace",
        bbox={"boxstyle": "round", "facecolor": "#f8fafc", "alpha": 0.9},
    )

    out_dir = os.path.join("results", "eval_logs", task_id)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{task_id}_phase{phase}_training_curves.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Training curves saved -> {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Track A training curves from monitor/TensorBoard artifacts")
    parser.add_argument("--task", required=True, choices=list(THRESHOLDS.keys()))
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2, 3])
    args = parser.parse_args()
    plot_training(task_id=args.task, phase=args.phase)


if __name__ == "__main__":
    main()
