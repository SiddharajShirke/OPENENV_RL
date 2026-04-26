#!/usr/bin/env python3
"""
OpenEnv baseline inference runner for Gov Workflow OpenEnv.

This script runs all 3 benchmark tasks (easy -> medium -> hard) and emits
strict, line-oriented stdout logs:

[START] task=<task_name> env=<benchmark> model=<model_name>
[STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI

from app.api_gateway import create_env_gateway
from app.baselines import backlog_clearance_policy
from app.models import ActionModel, ActionType, ObservationModel
from app.tasks import get_task

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None  # type: ignore[assignment]

if load_dotenv is not None:
    _ROOT = Path(__file__).resolve().parent
    load_dotenv(dotenv_path=_ROOT / ".env", override=False)

API_BASE_URL = os.getenv("API_BASE_URL", "https://integrate.api.nvidia.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta/llama-3.3-70b-instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
API_KEY = HF_TOKEN or OPENAI_API_KEY or os.getenv("API_KEY")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NVIDIA_API_KEY_2 = os.getenv("NVIDIA_API_KEY_2")
NVIDIA_MODEL = os.getenv("NVIDIA_MODEL", "")
ENV_TRANSPORT = os.getenv("OPENENV_ENV_TRANSPORT", "auto").strip().lower()
ENV_BASE_URL = os.getenv("OPENENV_ENV_BASE_URL", "http://127.0.0.1:7860").strip()
ENV_API_PREFIX = os.getenv("OPENENV_ENV_API_PREFIX", "").strip()
FORCE_FASTAPI_GATEWAY = os.getenv("FORCE_FASTAPI_GATEWAY", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

LEGACY_MODEL_POOL = [
    "meta/llama-3.3-70b-instruct",
    "qwen/qwen3-next-80b-a3b-instruct",
    "moonshotai/kimi-k2-instruct-0905",
    "meta/llama-3.1-405b-instruct",
    "deepseek-ai/deepseek-v3.2",
    "qwen/qwq-32b",
    "mistralai/mixtral-8x22b-instruct-v0.1",
    "google/gemma-3-27b-it",
    "microsoft/phi-4-mini-instruct",
    "meta/llama-3.1-8b-instruct",
]

BENCHMARK = "gov-workflow-openenv"
TASKS = [
    "district_backlog_easy",
    "mixed_urgency_medium",
    "cross_department_hard",
]
MAX_STEPS = int(os.getenv("MAX_STEPS", "80"))
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.50"))
TEMPERATURE = 0.0
MAX_TOKENS = 220

SYSTEM_PROMPT = (
    "You are controlling a government workflow environment. "
    "Return exactly one JSON object with these keys: "
    "action_type (required), and optional priority_mode, service, target_service, case_id, officer_delta. "
    "Allowed action_type: set_priority_mode, assign_capacity, request_missing_documents, "
    "escalate_service, advance_time, reallocate_officers. "
    "Allowed priority_mode: urgent_first, oldest_first, balanced, backlog_clearance. "
    "Allowed services: passport, driving_license, gst_registration, income_certificate, caste_certificate, "
    "birth_certificate, land_registration. "
    "Return lowercase values only and no explanation."
)


@dataclass
class EpisodeLog:
    rewards: list[float]
    steps: int
    score: float
    success: bool


@dataclass
class RuntimeContext:
    clients: list[OpenAI]
    model_pool: list[str]
    start_model_label: str


def _clean_token(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def _bool_str(value: bool) -> str:
    return "true" if value else "false"


def _sanitize_action_for_log(action: ActionModel) -> str:
    return json.dumps(action.model_dump(exclude_none=True), separators=(",", ":"))


def _sanitize_error_for_log(error: str | None) -> str:
    if not error:
        return "null"
    return error.replace("\n", " ").replace("\r", " ")


def _extract_json_object(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()
    if not text:
        return None

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None

    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

    return parsed if isinstance(parsed, dict) else None


def _coerce_action(payload: dict[str, Any] | None) -> ActionModel:
    if not payload:
        return ActionModel(action_type=ActionType.ADVANCE_TIME)

    norm = dict(payload)

    for key in ("action_type", "priority_mode", "service", "target_service"):
        if isinstance(norm.get(key), str):
            norm[key] = norm[key].strip().lower()

    if "officer_delta" in norm:
        try:
            norm["officer_delta"] = int(norm["officer_delta"])
        except (TypeError, ValueError):
            norm["officer_delta"] = 0

    try:
        return ActionModel(**norm)
    except Exception:
        return ActionModel(action_type=ActionType.ADVANCE_TIME)


def _build_user_prompt(task_id: str, step: int, observation: dict[str, Any], last_reward: float) -> str:
    compact_obs = json.dumps(observation, separators=(",", ":"))
    return (
        f"Task={task_id}. Step={step}. LastReward={last_reward:.2f}. "
        f"Observation={compact_obs}"
    )


def _choose_action(
    runtime: RuntimeContext,
    *,
    task_id: str,
    step: int,
    observation: ObservationModel,
    last_reward: float,
) -> ActionModel:
    prompt = _build_user_prompt(task_id, step, observation.model_dump(mode="json"), last_reward)

    for client in runtime.clients:
        for model_name in runtime.model_pool:
            try:
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    timeout=8.0,
                    stream=False,
                )
                content = (completion.choices[0].message.content or "").strip()
                action = _coerce_action(_extract_json_object(content))
                return action
            except Exception:
                # Try next model / key.
                continue

    # Final fallback when all API attempts fail or no API key exists.
    try:
        return backlog_clearance_policy(observation)
    except Exception:
        return ActionModel(action_type=ActionType.ADVANCE_TIME)


def _run_task(runtime: RuntimeContext, task_id: str) -> EpisodeLog:
    env = create_env_gateway(
        task_id=task_id,
        seed=get_task(task_id).seed,
        mode=ENV_TRANSPORT if ENV_TRANSPORT in {"auto", "http", "direct"} else "auto",
        base_url=ENV_BASE_URL,
        api_prefix=ENV_API_PREFIX,
        enforce_fastapi=FORCE_FASTAPI_GATEWAY,
    )
    print(f"[START] task={task_id} env={BENCHMARK} model={runtime.start_model_label}", flush=True)

    rewards: list[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    try:
        obs = env.reset()
        last_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            if env.terminated or env.truncated:
                break

            action = _choose_action(
                runtime,
                task_id=task_id,
                step=step,
                observation=obs,
                last_reward=last_reward,
            )

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            last_error = getattr(info, "last_action_message", None)

            rewards.append(float(reward))
            steps_taken = step
            last_reward = float(reward)

            print(
                f"[STEP] step={step} action={_sanitize_action_for_log(action)} "
                f"reward={reward:.2f} done={_bool_str(done)} "
                f"error={_sanitize_error_for_log(last_error)}",
                flush=True,
            )

            if done:
                break

        score, _grader_name, _metrics = env.grade()
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                pass

        rewards_str = ",".join(f"{r:.2f}" for r in rewards)
        print(
            f"[END] success={_bool_str(success)} steps={steps_taken} "
            f"score={score:.2f} rewards={rewards_str}",
            flush=True,
        )

    return EpisodeLog(rewards=rewards, steps=steps_taken, score=score, success=success)


def main() -> None:
    # LOCAL_IMAGE_NAME is read for compatibility with OpenEnv docker-based runners.
    _ = LOCAL_IMAGE_NAME
    keys: list[str] = []
    for k in (
        _clean_token(API_KEY),
        _clean_token(HF_TOKEN),
        _clean_token(OPENAI_API_KEY),
        _clean_token(os.getenv("API_KEY")),
        _clean_token(NVIDIA_API_KEY),
        _clean_token(NVIDIA_API_KEY_2),
    ):
        if k and k not in keys:
            keys.append(k)

    model_pool: list[str] = []
    for model_name in (MODEL_NAME, NVIDIA_MODEL, *LEGACY_MODEL_POOL):
        if model_name and model_name not in model_pool:
            model_pool.append(model_name)

    clients: list[OpenAI] = []
    for k in keys:
        try:
            clients.append(OpenAI(base_url=API_BASE_URL, api_key=k, max_retries=0, timeout=8.0))
        except Exception:
            continue

    start_model_label = model_pool[0] if clients else "local-heuristic-fallback"
    runtime = RuntimeContext(
        clients=clients,
        model_pool=model_pool,
        start_model_label=start_model_label,
    )

    for task_id in TASKS:
        _run_task(runtime, task_id)


if __name__ == "__main__":
    main()
