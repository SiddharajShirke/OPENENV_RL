"""
app/story_router.py

FastAPI router that serves LLM training story data.
All 7 endpoints are READ-ONLY - they serve pre-saved JSON files.
No frontend elements are invoked from backend.
No training runs happen here - only data serving.

Mount in main.py with:
  from app.story_router import router as story_router
  app.include_router(story_router)
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

router = APIRouter(prefix="/training", tags=["Training Story"])

# --- Data directory --------------------------------------------------
DATA_DIR = Path("data/training_logs")

HEURISTIC_BASELINES: dict[str, dict] = {
    "district_backlog_easy": {
        "score": 0.527, "completed": 41,
        "breaches": 184, "reward": -79.86, "avg_wait": 6.9,
    },
    "mixed_urgency_medium": {
        "score": 0.454, "completed": 58,
        "breaches": 34,  "reward": -684.22, "avg_wait": 12.4,
    },
    "cross_department_hard": {
        "score": 0.606, "completed": 83,
        "breaches": 723, "reward": -2318.78, "avg_wait": 15.6,
    },
}


# --- Internal helpers ------------------------------------------------

def _load_log(task_id: str) -> dict:
    """Load JSON training log for given task. Raises 404 if missing."""
    path = DATA_DIR / f"{task_id}_training_log.json"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"Training log not found for task '{task_id}'. "
                f"Run: python scripts/convert_grpo_csv.py "
                f"--csv <your_csv> --task {task_id}"
            ),
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _dominant_action(episodes: list[dict]) -> str:
    """Returns the action name with the highest total weight across episodes."""
    totals: dict[str, float] = {}
    for ep in episodes:
        for action, val in ep.get("actions", {}).items():
            totals[action] = totals.get(action, 0.0) + float(val)
    return max(totals, key=totals.get) if totals else "advance_time"


def _phase_message(ep: dict) -> str:
    """Returns a human-readable learning message for one episode."""
    phase = ep.get("phase", "random")
    reward = ep.get("total_reward", 0)
    score = ep.get("score", 0)
    fn1 = ep.get("fn1_valid", 1.0)
    fn2 = ep.get("fn2_no_halluc", 1.0)
    episode = ep.get("episode", 0)

    validity_note = "" if fn1 >= 1.0 else f" WARNING: Invalid action at step {episode}."
    halluc_note = "" if fn2 >= 1.0 else " WARNING: Hallucination detected."

    messages = {
        "random": (
            f"Step {episode}: LLM is exploring. "
            f"Reward={reward:.3f}, Score={score:.3f}.{validity_note}{halluc_note}"
        ),
        "exploring": (
            f"Step {episode}: LLM finding patterns. "
            f"Reward={reward:.3f}, Score={score:.3f}.{validity_note}{halluc_note}"
        ),
        "learning": (
            f"Step {episode}: LLM reinforcing good actions. "
            f"Reward={reward:.3f}, Score={score:.3f}.{validity_note}{halluc_note}"
        ),
        "converged": (
            f"Step {episode}: LLM converged. "
            f"Reward={reward:.3f}, Score={score:.3f}.{validity_note}{halluc_note}"
        ),
    }
    return messages.get(phase, f"Step {episode}: reward={reward:.3f}")


# ================================================================
# ENDPOINT 1 - GET /training/tasks
# ================================================================
@router.get("/tasks")
async def list_trained_tasks() -> dict:
    """
    Returns all tasks that have a saved training log JSON file.
    Frontend calls this first to populate task selector.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    available = []
    for path in sorted(DATA_DIR.glob("*_training_log.json")):
        task_id = path.stem.replace("_training_log", "")
        try:
            log = _load_log(task_id)
            available.append({
                "task_id":            task_id,
                "total_episodes":     log["total_episodes"],
                "final_score":        log["summary"]["last_episode_score"],
                "reward_improvement": log["summary"]["reward_improvement_pct"],
                "base_model":         log.get("base_model", ""),
                "training_method":    log.get("training_method", "GRPO"),
            })
        except HTTPException:
            pass
    return {"tasks": available}


# ================================================================
# ENDPOINT 2 - GET /training/summary/{task_id}
# ================================================================
@router.get("/summary/{task_id}")
async def training_summary(task_id: str) -> dict:
    """Returns overview stats + narrative for the ACT 2 header card."""
    log = _load_log(task_id)
    eps = log["episodes"]
    n = len(eps)

    q1, q2, q3 = n // 4, n // 2, 3 * n // 4

    p1_dom = _dominant_action(eps[:q1])
    p2_dom = _dominant_action(eps[q1:q2])
    p3_dom = _dominant_action(eps[q2:q3])
    p4_dom = _dominant_action(eps[q3:])

    avg_p1_r = sum(e["total_reward"] for e in eps[:q1]) / max(q1, 1)
    avg_p4_r = sum(e["total_reward"] for e in eps[q3:]) / max(n - q3, 1)

    return {
        "task_id":          log["task_id"],
        "base_model":       log.get("base_model", ""),
        "training_method":  log.get("training_method", "GRPO"),
        "lora_rank":        log.get("lora_rank", 16),
        "total_episodes":   n,
        "reward_functions": log.get("reward_functions", {}),
        "summary":          log["summary"],
        "narrative": {
            "phase_1": (
                f"Steps 1-{q1}: LLM chose '{p1_dom}' most often. "
                f"Avg reward {avg_p1_r:.2f}. Still exploring randomly."
            ),
            "phase_2": (
                f"Steps {q1}-{q2}: LLM discovered '{p2_dom}'. "
                "Reward started improving as valid patterns emerged."
            ),
            "phase_3": (
                f"Steps {q2}-{q3}: LLM reinforced '{p3_dom}'. "
                "Action validity reaching near-perfect levels."
            ),
            "phase_4": (
                f"Steps {q3}-{n}: LLM converged on '{p4_dom}'. "
                f"Avg reward {avg_p4_r:.2f}. "
                f"Final score {log['summary']['last_episode_score']:.1%}."
            ),
        },
    }


# ================================================================
# ENDPOINT 3 - GET /training/curve/{task_id}
# ================================================================
@router.get("/curve/{task_id}")
async def training_curve(
    task_id: str,
    downsample: int = 1,
) -> dict:
    """
    Returns episode-by-episode reward + score for chart rendering.
    downsample=5 -> returns every 5th step.
    """
    log = _load_log(task_id)
    eps = log["episodes"]
    sampled = eps[::max(1, downsample)]
    return {
        "task_id":      task_id,
        "total_points": len(sampled),
        "curve": [
            {
                "episode":       e["episode"],
                "reward":        e["total_reward"],
                "score":         e["score"],
                "fn1_valid":     e.get("fn1_valid", 1.0),
                "fn2_no_halluc": e.get("fn2_no_halluc", 1.0),
                "fn3_env_score": e.get("fn3_env_score", 0.0),
                "phase":         e["phase"],
            }
            for e in sampled
        ],
    }


# ================================================================
# ENDPOINT 4 - GET /training/actions/{task_id}
# ================================================================
@router.get("/actions/{task_id}")
async def action_evolution(task_id: str) -> dict:
    """Returns action distribution at 5 checkpoints across training."""
    log = _load_log(task_id)
    eps = log["episodes"]
    n = len(eps)

    idxs = [0, n // 4, n // 2, 3 * n // 4, n - 1]
    result = []
    for idx in idxs:
        ep = eps[idx]
        result.append({
            "episode": ep["episode"],
            "phase":   ep["phase"],
            "actions": ep.get("actions", {}),
            "reward":  ep["total_reward"],
            "score":   ep["score"],
        })

    avg_fn1_start = sum(e.get("fn1_valid", 1.0) for e in eps[:n // 4]) / max(n // 4, 1)
    avg_fn1_end = sum(e.get("fn1_valid", 1.0) for e in eps[3 * n // 4:]) / max(n - 3 * n // 4, 1)

    insight = (
        f"Action validity improved from {avg_fn1_start:.1%} (early) "
        f"to {avg_fn1_end:.1%} (final). "
        "LLM learned to output valid government workflow JSON consistently."
    )

    return {
        "task_id":     task_id,
        "checkpoints": result,
        "insight":     insight,
    }


# ================================================================
# ENDPOINT 5 - GET /training/episode/{task_id}/{episode_num}
# ================================================================
@router.get("/episode/{task_id}/{episode_num}")
async def episode_detail(task_id: str, episode_num: int) -> dict:
    """Returns detail for one specific training step."""
    log = _load_log(task_id)
    eps = log["episodes"]

    if episode_num < 1 or episode_num > len(eps):
        raise HTTPException(
            status_code=400,
            detail=f"episode_num must be 1-{len(eps)}. Got {episode_num}.",
        )

    ep = eps[episode_num - 1]
    rewards_so_far = [e["total_reward"] for e in eps[:episode_num]]
    scores_so_far = [e["score"] for e in eps[:episode_num]]

    return {
        "task_id":             task_id,
        "episode":             ep["episode"],
        "total_episodes":      len(eps),
        "reward":              ep["total_reward"],
        "score":               ep["score"],
        "fn1_valid":           ep.get("fn1_valid", 1.0),
        "fn2_no_halluc":       ep.get("fn2_no_halluc", 1.0),
        "fn3_env_score":       ep.get("fn3_env_score", 0.0),
        "phase":               ep["phase"],
        "actions":             ep.get("actions", {}),
        "running_best_reward": max(rewards_so_far),
        "running_avg_score":   round(sum(scores_so_far) / len(scores_so_far), 4),
        "message":             _phase_message(ep),
    }


# ================================================================
# ENDPOINT 6 - GET /training/stream/{task_id} [SSE]
# ================================================================
@router.get("/stream/{task_id}")
async def stream_training_replay(
    task_id: str,
    delay_ms: int = 100,
    start_episode: int = 1,
    end_episode: Optional[int] = None,
) -> StreamingResponse:
    """Server-Sent Events endpoint for animated chart replay."""
    log = _load_log(task_id)
    eps = log["episodes"]
    end = min(end_episode or len(eps), len(eps))
    subset = eps[start_episode - 1: end]

    async def generate():
        meta_event = json.dumps({
            "type":             "meta",
            "task_id":          task_id,
            "total_episodes":   len(eps),
            "summary":          log["summary"],
            "reward_functions": log.get("reward_functions", {}),
        })
        yield f"data: {meta_event}\n\n"

        rewards_so_far: list[float] = []
        scores_so_far: list[float] = []

        for ep in subset:
            rewards_so_far.append(ep["total_reward"])
            scores_so_far.append(ep["score"])

            event = json.dumps({
                "type":              "episode",
                "episode":           ep["episode"],
                "total_episodes":    len(eps),
                "reward":            ep["total_reward"],
                "score":             ep["score"],
                "fn1_valid":         ep.get("fn1_valid",     1.0),
                "fn2_no_halluc":     ep.get("fn2_no_halluc", 1.0),
                "fn3_env_score":     ep.get("fn3_env_score", 0.0),
                "phase":             ep["phase"],
                "actions":           ep.get("actions", {}),
                "running_best":      max(rewards_so_far),
                "running_avg_score": round(
                    sum(scores_so_far) / len(scores_so_far), 4
                ),
                "message":           _phase_message(ep),
            })
            yield f"data: {event}\n\n"
            await asyncio.sleep(delay_ms / 1000.0)

        done_event = json.dumps({
            "type":        "done",
            "final_score": scores_so_far[-1] if scores_so_far else 0.0,
            "best_reward": max(rewards_so_far) if rewards_so_far else 0.0,
            "total_steps": len(subset),
        })
        yield f"data: {done_event}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ================================================================
# ENDPOINT 7 - GET /training/comparison/{task_id}
# ================================================================
@router.get("/comparison/{task_id}")
async def before_after_comparison(task_id: str) -> dict:
    """Returns before (heuristic) vs after (trained LLM)."""
    log = _load_log(task_id)
    baseline = HEURISTIC_BASELINES.get(task_id, {})
    summary = log["summary"]

    bef_score = baseline.get("score", 0.0)
    after_score = summary["last_episode_score"]
    delta = round(after_score - bef_score, 4)
    pct = round((delta / bef_score) * 100, 1) if bef_score else 0.0

    return {
        "task_id": task_id,
        "before": {
            "label":     "Heuristic Baseline (no AI)",
            "score":     bef_score,
            "reward":    baseline.get("reward",    0.0),
            "completed": baseline.get("completed", 0),
            "breaches":  baseline.get("breaches",  0),
            "avg_wait":  baseline.get("avg_wait",  0.0),
        },
        "after": {
            "label":               f"GRPO Trained LLM ({log.get('base_model','')})",
            "score":               after_score,
            "reward":              summary["last_episode_reward"],
            "avg_fn1_valid":       summary.get("avg_fn1_valid",     0.0),
            "avg_fn2_no_halluc":   summary.get("avg_fn2_no_halluc", 0.0),
            "invalid_steps":       summary.get("invalid_action_steps", 0),
            "hallucination_steps": summary.get("hallucination_steps",  0),
        },
        "improvement": {
            "score_delta": delta,
            "score_pct":   pct,
            "verdict": (
                "LLM significantly outperforms baseline"
                if delta > 0.10 else
                "LLM moderately outperforms baseline"
                if delta > 0.0 else
                "LLM needs more training"
            ),
        },
    }
