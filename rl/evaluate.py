"""
Deterministic evaluator: runs a trained model on tasks and returns grader scores.

Usage:
    python -m rl.evaluate --model results/best_model/phase2_final.zip --episodes 3
    python -m rl.evaluate --model results/best_model/phase3_final.zip --episodes 3 --model-type recurrent
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from typing import Any, Literal

import numpy as np
from sb3_contrib import MaskablePPO, RecurrentPPO
from sb3_contrib.common.maskable.utils import get_action_masks

from rl.gov_workflow_env import GovWorkflowGymEnv
from app.graders import grade_episode
from app.tasks import TASKS

TASK_IDS = [
    "district_backlog_easy",
    "mixed_urgency_medium",
    "cross_department_hard",
]

ModelType = Literal["auto", "maskable", "recurrent"]


@dataclass
class TaskEvalResult:
    task_id: str
    seed: int
    grader_score: float
    total_reward: float
    total_steps: int
    total_completed: int
    total_sla_breaches: int
    fairness_gap: float


def _normalize_action(action: Any) -> int:
    if isinstance(action, np.ndarray):
        return int(action.item())
    return int(action)


def _apply_eval_action_mask(action_idx: int, masks: np.ndarray) -> int:
    if 0 <= action_idx < masks.shape[0] and bool(masks[action_idx]):
        return action_idx
    if masks.shape[0] > 18 and bool(masks[18]):
        return 18
    valid = np.flatnonzero(masks)
    if valid.size == 0:
        return 18
    return int(valid[0])


def predict_recurrent_action(
    model: Any,
    obs: np.ndarray,
    lstm_state: Any,
    episode_start: np.ndarray,
    masks: np.ndarray,
) -> tuple[int, Any]:
    action, next_state = model.predict(
        obs,
        state=lstm_state,
        episode_start=episode_start,
        deterministic=True,
    )
    action_idx = _normalize_action(action)
    action_idx = _apply_eval_action_mask(action_idx, masks)
    return action_idx, next_state


def _load_model(model_path: str, model_type: ModelType) -> tuple[Any, str]:
    if model_type == "maskable":
        try:
            return MaskablePPO.load(model_path), "maskable"
        except Exception as exc:
            raise ValueError(
                "Failed to load as MaskablePPO. This checkpoint may be recurrent. "
                "Try: --model-type recurrent"
            ) from exc
    if model_type == "recurrent":
        try:
            return RecurrentPPO.load(model_path), "recurrent"
        except Exception as exc:
            raise ValueError(
                "Failed to load as RecurrentPPO. This checkpoint may be maskable. "
                "Try: --model-type maskable"
            ) from exc

    try:
        return MaskablePPO.load(model_path), "maskable"
    except Exception:
        return RecurrentPPO.load(model_path), "recurrent"


def evaluate_model(
    model_path: str,
    task_ids: list[str] = TASK_IDS,
    n_episodes: int = 1,
    verbose: bool = True,
    model_type: ModelType = "auto",
) -> list[TaskEvalResult]:
    model, resolved_type = _load_model(model_path, model_type)
    results = []

    for task_id in task_ids:
        task_cfg = TASKS.get(task_id)
        if task_cfg is None:
            print(f"[Eval] Task {task_id!r} not found, skipping.")
            continue

        ep_rewards, ep_scores = [], []
        last_info: dict[str, Any] = {}

        for ep in range(n_episodes):
            env = GovWorkflowGymEnv(task_id=task_id, seed=task_cfg.seed + ep)
            obs, _ = env.reset()
            done, ep_reward = False, 0.0

            if resolved_type == "recurrent":
                lstm_state: Any = None
                episode_start = np.array([True], dtype=bool)

                while not done:
                    masks = env.action_masks()
                    action_idx, lstm_state = predict_recurrent_action(
                        model=model,
                        obs=obs,
                        lstm_state=lstm_state,
                        episode_start=episode_start,
                        masks=masks,
                    )
                    obs, reward, terminated, truncated, info = env.step(action_idx)
                    ep_reward += reward
                    done = terminated or truncated
                    episode_start = np.array([done], dtype=bool)
                    last_info = info
            else:
                while not done:
                    masks = get_action_masks(env)
                    action, _ = model.predict(obs, action_masks=masks, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(int(action))
                    ep_reward += reward
                    done = terminated or truncated
                    last_info = info

            gr = grade_episode(env._core_env.state())
            ep_rewards.append(ep_reward)
            ep_scores.append(gr.score)

        ep_state = env._core_env.state()
        result = TaskEvalResult(
            task_id=task_id,
            seed=task_cfg.seed,
            grader_score=float(np.mean(ep_scores)),
            total_reward=float(np.mean(ep_rewards)),
            total_steps=ep_state.total_steps,
            total_completed=ep_state.total_completed,
            total_sla_breaches=ep_state.total_sla_breaches,
            fairness_gap=float(last_info.get("fairness_gap", 0.0)),
        )
        results.append(result)
        if verbose:
            print(
                f"[Eval] {task_id:<30} "
                f"score={result.grader_score:.4f}  "
                f"reward={result.total_reward:.2f}  "
                f"completed={result.total_completed}  "
                f"sla_breaches={result.total_sla_breaches}"
            )
    return results


def compare_recurrent_vs_flat(
    flat_model_path: str,
    recurrent_model_path: str,
    task_id: str = "mixed_urgency_medium",
    n_episodes: int = 3,
) -> dict[str, float]:
    flat = evaluate_model(
        flat_model_path,
        task_ids=[task_id],
        n_episodes=n_episodes,
        verbose=False,
        model_type="maskable",
    )[0].grader_score
    recurrent = evaluate_model(
        recurrent_model_path,
        task_ids=[task_id],
        n_episodes=n_episodes,
        verbose=False,
        model_type="recurrent",
    )[0].grader_score
    return {
        "flat": float(flat),
        "recurrent": float(recurrent),
        "delta": float(recurrent - flat),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO model")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--task",
        default=None,
        choices=TASK_IDS,
        help="Single-task alias. If set, overrides --tasks.",
    )
    parser.add_argument("--tasks", nargs="+", default=TASK_IDS)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--model-type",
        choices=["auto", "maskable", "recurrent"],
        default="auto",
        help="Model class to load. Use auto for best-effort detection.",
    )
    args = parser.parse_args()

    selected_tasks = [args.task] if args.task else args.tasks
    results = evaluate_model(
        args.model,
        task_ids=selected_tasks,
        n_episodes=args.episodes,
        model_type=args.model_type,
    )
    if args.output:
        import os

        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        print(f"\n[Eval] Results saved to {args.output}")

    avg = np.mean([r.grader_score for r in results])
    print(f"\n[Eval] Average grader score: {avg:.4f}")


if __name__ == "__main__":
    main()
