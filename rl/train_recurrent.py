"""
Phase 3: Recurrent PPO (LSTM policy) training.

This trainer keeps the existing 28-action design and uses curriculum sampling
across tasks (easy -> medium -> hard). Because current sb3-contrib releases do
not provide MaskableRecurrentPPO, we enforce action masks in two places:
1) hard mask in GovWorkflowGymEnv before executing an action,
2) recurrent evaluation callback with masked action sanitization.

Usage:
    python -m rl.train_recurrent --timesteps 600000
"""

from __future__ import annotations

import argparse
import os
from typing import Any

import yaml
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import MaskablePPO, RecurrentPPO

from rl.callbacks import CostMonitorCallback, RecurrentEvalCallback
from rl.curriculum import CurriculumConfig, CurriculumScheduler
from rl.gov_workflow_env import GovWorkflowGymEnv

os.makedirs("results/runs", exist_ok=True)
os.makedirs("results/best_model", exist_ok=True)
os.makedirs("results/eval_logs", exist_ok=True)


def _load_cfg(path: str) -> dict:
    if os.path.exists(path):
        with open(path, encoding="utf-8-sig") as f:
            return yaml.safe_load(f)
    return {}


def _transfer_matching_policy_weights(
    recurrent_model: RecurrentPPO,
    flat_model_path: str,
    exclude_prefixes: tuple[str, ...] = (),
) -> int:
    """
    Transfer compatible policy weights from a flat MaskablePPO checkpoint.

    Returns number of copied tensors.
    """
    src_path = flat_model_path
    if not src_path.endswith(".zip"):
        src_path = f"{src_path}.zip"
    if not os.path.exists(src_path):
        return 0

    try:
        flat_model = MaskablePPO.load(src_path)
    except Exception as exc:
        print(f"[Phase 3] Skipping flat-weight transfer, could not load MaskablePPO from {src_path}: {exc}")
        return 0
    src_state = flat_model.policy.state_dict()
    dst_state = recurrent_model.policy.state_dict()

    copied = 0
    for key, dst_tensor in dst_state.items():
        if any(key.startswith(prefix) for prefix in exclude_prefixes):
            continue
        src_tensor = src_state.get(key)
        if src_tensor is None:
            continue
        if tuple(src_tensor.shape) != tuple(dst_tensor.shape):
            continue
        dst_state[key] = src_tensor
        copied += 1

    recurrent_model.policy.load_state_dict(dst_state, strict=False)
    return copied


def train_phase3(
    total_timesteps: int = 600_000,
    n_envs: int = 4,
    seed: int = 42,
    config_path: str = "rl/configs/recurrent.yaml",
) -> RecurrentPPO:
    cfg = _load_cfg(config_path)
    hp = cfg.get("hyperparameters", {})
    cur_c = cfg.get("curriculum", {})
    tr_c = cfg.get("training", {})

    scheduler = CurriculumScheduler(
        total_timesteps=total_timesteps,
        config=CurriculumConfig(
            stage1_end_frac=float(cur_c.get("stage1_end_frac", 0.20)),
            stage2_end_frac=float(cur_c.get("stage2_end_frac", 0.55)),
            stage3_weights=tuple(cur_c.get("stage3_weights", [0.15, 0.35, 0.50])),
        ),
        rng_seed=seed,
    )

    global_step_counter = [0]
    hard_action_mask_train = bool(tr_c.get("hard_action_mask_train", True))
    hard_action_mask_eval = bool(tr_c.get("hard_action_mask_eval", True))

    def _sample_task() -> str:
        return scheduler.sample_task(global_step_counter[0])

    def _make_curr(rank: int):
        def _init():
            env = GovWorkflowGymEnv(
                task_id="district_backlog_easy",
                seed=seed + rank,
                hard_action_mask=hard_action_mask_train,
            )
            env.set_task_sampler(_sample_task, global_step_counter)
            return Monitor(env)

        return _init

    train_env = DummyVecEnv([_make_curr(i) for i in range(n_envs)])

    eval_task_id = str(tr_c.get("eval_task_id", "mixed_urgency_medium"))
    eval_env = GovWorkflowGymEnv(eval_task_id, seed=seed + 999, hard_action_mask=hard_action_mask_eval)

    eval_cb = RecurrentEvalCallback(
        eval_env=eval_env,
        eval_freq=int(tr_c.get("eval_freq", max(4096 // n_envs, 1))),
        n_eval_episodes=int(tr_c.get("n_eval_episodes", 3)),
        best_model_save_path="results/best_model",
        log_path="results/eval_logs",
        task_id=eval_task_id,
        verbose=1,
    )

    model = RecurrentPPO(
        policy="MlpLstmPolicy",
        env=train_env,
        learning_rate=float(hp.get("learning_rate", 1e-4)),
        n_steps=int(hp.get("n_steps", 512)),
        batch_size=int(hp.get("batch_size", 128)),
        n_epochs=int(hp.get("n_epochs", 10)),
        gamma=float(hp.get("gamma", 0.995)),
        gae_lambda=float(hp.get("gae_lambda", 0.95)),
        clip_range=float(hp.get("clip_range", 0.2)),
        ent_coef=float(hp.get("ent_coef", 0.002)),
        vf_coef=float(hp.get("vf_coef", 0.5)),
        max_grad_norm=float(hp.get("max_grad_norm", 0.5)),
        policy_kwargs=dict(
            net_arch=hp.get("net_arch", [256, 256]),
            lstm_hidden_size=int(hp.get("lstm_hidden_size", 128)),
            n_lstm_layers=int(hp.get("n_lstm_layers", 1)),
            shared_lstm=bool(hp.get("shared_lstm", False)),
            enable_critic_lstm=bool(hp.get("enable_critic_lstm", True)),
        ),
        tensorboard_log="results/runs/phase3_recurrent_ppo",
        verbose=1,
        seed=seed,
    )

    warm_start_from = str(tr_c.get("warm_start_from", "results/best_model/phase2_final"))
    transfer_flat = bool(tr_c.get("transfer_flat_weights", True))
    transfer_exclude_prefixes = tuple(
        tr_c.get("transfer_exclude_prefixes", ["action_net.", "value_net."])
    )
    if transfer_flat:
        copied = _transfer_matching_policy_weights(
            model,
            warm_start_from,
            exclude_prefixes=transfer_exclude_prefixes,
        )
        if copied > 0:
            print(f"[Phase 3] Transferred {copied} compatible policy tensors from {warm_start_from}")
        else:
            print(f"[Phase 3] No compatible transfer tensors found from {warm_start_from}")

    print(f"\n[Phase 3] Recurrent PPO | timesteps={total_timesteps} | n_envs={n_envs}")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_cb, CostMonitorCallback()],
        tb_log_name="recurrent_ppo",
        progress_bar=True,
    )
    model.save("results/best_model/phase3_final")
    print("[Phase 3] Done -> results/best_model/phase3_final")
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=600_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", default="rl/configs/recurrent.yaml")
    args = parser.parse_args()

    train_phase3(
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        seed=args.seed,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
