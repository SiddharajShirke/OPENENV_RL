"""
Phase 1: Masked PPO on district_backlog_easy.
Phase 2: Curriculum Masked PPO across all 3 tasks.

Usage:
    python -m rl.train_ppo --phase 1 --timesteps 200000
    python -m rl.train_ppo --phase 2 --timesteps 500000
"""

from __future__ import annotations

import argparse
import os

import yaml
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import MaskablePPO

from rl.gym_wrapper import GovWorkflowGymEnv
from rl.callbacks import GovWorkflowEvalCallback, CostMonitorCallback
from rl.curriculum import CurriculumScheduler, CurriculumConfig

os.makedirs("results/runs",       exist_ok=True)
os.makedirs("results/best_model", exist_ok=True)
os.makedirs("results/eval_logs",  exist_ok=True)


def _load_cfg(path: str) -> dict:
    if os.path.exists(path):
        # `utf-8-sig` safely handles files with/without UTF-8 BOM.
        with open(path, encoding="utf-8-sig") as f:
            return yaml.safe_load(f)
    return {}


# ---------------------------------------------------------------------------
# Phase 1 — single task easy
# ---------------------------------------------------------------------------
def train_phase1(
    total_timesteps: int = 200_000,
    n_envs:          int = 4,
    seed:            int = 42,
    config_path:     str = "rl/configs/ppo_easy.yaml",
) -> MaskablePPO:
    cfg = _load_cfg(config_path)
    hp  = cfg.get("hyperparameters", {})

    def _make(rank: int):
        def _init():
            return Monitor(GovWorkflowGymEnv("district_backlog_easy", seed=seed + rank))
        return _init

    train_env = DummyVecEnv([_make(i) for i in range(n_envs)])
    eval_env  = GovWorkflowGymEnv("district_backlog_easy", seed=seed + 1000)

    eval_cb = GovWorkflowEvalCallback(
        eval_env=eval_env,
        eval_freq=max(2048 // n_envs, 1),
        n_eval_episodes=5,
        best_model_save_path="results/best_model",
        log_path="results/eval_logs",
        task_id="district_backlog_easy",
        verbose=1,
    )

    model = MaskablePPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=float(hp.get("learning_rate", 3e-4)),
        n_steps=int(hp.get("n_steps",     512)),
        batch_size=int(hp.get("batch_size", 64)),
        n_epochs=int(hp.get("n_epochs",   10)),
        gamma=float(hp.get("gamma",        0.99)),
        gae_lambda=float(hp.get("gae_lambda", 0.95)),
        clip_range=float(hp.get("clip_range", 0.2)),
        ent_coef=float(hp.get("ent_coef",     0.01)),
        vf_coef=float(hp.get("vf_coef",       0.5)),
        max_grad_norm=float(hp.get("max_grad_norm", 0.5)),
        policy_kwargs=dict(net_arch=hp.get("net_arch", [256, 256])),
        tensorboard_log="results/runs/phase1_masked_ppo",
        verbose=1,
        seed=seed,
    )

    print(f"\n[Phase 1] Masked PPO | timesteps={total_timesteps} | n_envs={n_envs}")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_cb, CostMonitorCallback()],
        tb_log_name="masked_ppo_easy",
        progress_bar=True,
    )
    model.save("results/best_model/phase1_final")
    print("[Phase 1] Done -> results/best_model/phase1_final")
    return model


# ---------------------------------------------------------------------------
# Phase 2 — curriculum across all tasks
# ---------------------------------------------------------------------------
def train_phase2(
    total_timesteps: int = 500_000,
    n_envs:          int = 4,
    seed:            int = 42,
    config_path:     str = "rl/configs/curriculum.yaml",
) -> MaskablePPO:
    cfg = _load_cfg(config_path)
    if not cfg and config_path.endswith("curriculum.yaml"):
        # Backward compatibility with previous filename.
        cfg = _load_cfg("rl/configs/ppo_curriculum.yaml")
    hp    = cfg.get("hyperparameters", {})
    cur_c = cfg.get("curriculum", {})
    tr_c  = cfg.get("training", {})

    scheduler = CurriculumScheduler(
        total_timesteps=total_timesteps,
        config=CurriculumConfig(
            stage1_end_frac=float(cur_c.get("stage1_end_frac", 0.30)),
            stage2_end_frac=float(cur_c.get("stage2_end_frac", 0.70)),
            stage3_weights=tuple(cur_c.get("stage3_weights", [0.20, 0.40, 0.40])),
        ),
        rng_seed=seed,
    )

    global_step_counter = [0]

    def _sample_task() -> str:
        return scheduler.sample_task(global_step_counter[0])

    def _make_curr(rank: int):
        def _init():
            env = GovWorkflowGymEnv(
                task_id="district_backlog_easy",
                seed=seed + rank,
            )
            env.set_task_sampler(_sample_task, global_step_counter)
            return Monitor(env)
        return _init

    train_env = DummyVecEnv([_make_curr(i) for i in range(n_envs)])
    eval_task_id = str(tr_c.get("eval_task_id", "mixed_urgency_medium"))
    eval_env  = GovWorkflowGymEnv(eval_task_id, seed=seed + 999)

    eval_cb = GovWorkflowEvalCallback(
        eval_env=eval_env,
        eval_freq=int(tr_c.get("eval_freq", max(4096 // n_envs, 1))),
        n_eval_episodes=int(tr_c.get("n_eval_episodes", 3)),
        best_model_save_path="results/best_model",
        log_path="results/eval_logs",
        task_id=eval_task_id,
        verbose=1,
    )

    warm_start_from = str(tr_c.get("warm_start_from", "results/best_model/phase1_final"))
    warm_start_path = warm_start_from
    if not warm_start_path.endswith(".zip"):
        warm_start_path = f"{warm_start_path}.zip"

    if os.path.exists(warm_start_path):
        print(f"[Phase 2] Warm-starting from {warm_start_path}")
        model = MaskablePPO.load(warm_start_path, env=train_env)
    else:
        model = MaskablePPO(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=float(hp.get("learning_rate", 2e-4)),
            n_steps=int(hp.get("n_steps",     512)),
            batch_size=int(hp.get("batch_size", 64)),
            n_epochs=int(hp.get("n_epochs",    10)),
            gamma=float(hp.get("gamma",         0.99)),
            gae_lambda=float(hp.get("gae_lambda",  0.95)),
            clip_range=float(hp.get("clip_range",  0.2)),
            ent_coef=float(hp.get("ent_coef",      0.005)),
            policy_kwargs=dict(net_arch=hp.get("net_arch", [256, 256])),
            tensorboard_log="results/runs/phase2_curriculum_ppo",
            verbose=1,
            seed=seed,
        )

    print(f"\n[Phase 2] Curriculum PPO | timesteps={total_timesteps}")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_cb, CostMonitorCallback()],
        tb_log_name="curriculum_ppo",
        progress_bar=True,
    )
    model.save("results/best_model/phase2_final")
    print("[Phase 2] Done -> results/best_model/phase2_final")
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase",     type=int, choices=[1, 2], default=1)
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--n-envs",    type=int, default=4)
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument(
        "--phase1-config",
        default="rl/configs/ppo_easy.yaml",
        help="Config file for Phase 1 training.",
    )
    parser.add_argument(
        "--phase2-config",
        default="rl/configs/curriculum.yaml",
        help="Config file for Phase 2 curriculum training.",
    )
    args = parser.parse_args()

    if args.phase == 1:
        train_phase1(
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
            seed=args.seed,
            config_path=args.phase1_config,
        )
    else:
        train_phase2(
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
            seed=args.seed,
            config_path=args.phase2_config,
        )


if __name__ == "__main__":
    main()
