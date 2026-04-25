"""
Custom SB3 callbacks for Gov Workflow RL training.

GovWorkflowEvalCallback  -- MaskableEvalCallback + grader score logging
CostMonitorCallback      -- per-rollout cost constraint logging to TensorBoard
"""

from __future__ import annotations

import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from typing import Any

from rl.gov_workflow_env import GovWorkflowGymEnv
from rl.cost_tracker import THRESHOLD_SLA, THRESHOLD_FAIRNESS


class GovWorkflowEvalCallback(MaskableEvalCallback):
    """
    Extends MaskableEvalCallback:
    1. Runs the deterministic grader after each eval.
    2. Logs grader score to TensorBoard.
    3. Saves best model by grader score (not just mean reward).
    """

    def __init__(
        self,
        eval_env:             GovWorkflowGymEnv,
        eval_freq:            int = 2048,
        n_eval_episodes:      int = 5,
        grader_eval_freq_multiplier: int = 4,
        grader_eval_max_steps: int | None = None,
        best_model_save_path: str = "results/best_model",
        log_path:             str = "results/eval_logs",
        task_id:              str = "district_backlog_easy",
        verbose:              int = 1,
    ):
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            best_model_save_path=best_model_save_path,
            log_path=log_path,
            verbose=verbose,
            warn=False,
        )
        self.task_id = task_id
        self.grader_eval_freq_multiplier = max(1, int(grader_eval_freq_multiplier))
        self.grader_eval_max_steps = grader_eval_max_steps
        self._best_grader_score = -np.inf
        os.makedirs(best_model_save_path, exist_ok=True)
        os.makedirs(log_path, exist_ok=True)

    def _on_step(self) -> bool:
        result = super()._on_step()
        grader_eval_freq = max(self.eval_freq * self.grader_eval_freq_multiplier, 1)
        if self.eval_freq > 0 and self.n_calls % grader_eval_freq == 0:
            grader_score = self._run_grader_eval()
            if self.logger:
                self.logger.record("eval/grader_score", grader_score)
            if grader_score > self._best_grader_score:
                self._best_grader_score = grader_score
                save_path = os.path.join(
                    self.best_model_save_path, f"best_grader_{self.task_id}"
                )
                self.model.save(save_path)
                if self.verbose:
                    print(f"[Eval] New best grader score: {grader_score:.4f} -> {save_path}")
        return result

    def _run_grader_eval(self) -> float:
        try:
            from app.graders import grade_episode
            from app.tasks import TASKS
            task_cfg = TASKS.get(self.task_id)
            if task_cfg is None:
                return 0.0
            max_steps = (
                int(self.grader_eval_max_steps)
                if self.grader_eval_max_steps is not None
                else max(1, int(task_cfg.max_days) * 10)
            )
            env = GovWorkflowGymEnv(task_id=self.task_id, seed=task_cfg.seed, hard_action_mask=True)
            obs, _ = env.reset()
            done = False
            steps = 0
            while not done:
                masks = np.asarray(env.action_masks(), dtype=bool).reshape(-1)
                action, _ = self.model.predict(obs, action_masks=masks, deterministic=True)
                obs, _, terminated, truncated, _ = env.step(int(action))
                done = terminated or truncated
                steps += 1
                if steps >= max_steps and not done:
                    break
            result = grade_episode(env._core_env.state())
            return float(result.score)
        except Exception as e:
            if self.verbose:
                print(f"[Eval] Grader eval failed: {e}")
            return 0.0


class CostMonitorCallback(BaseCallback):
    """
    Monitors SLA and fairness cost signals per rollout.
    Phase 1-3: diagnostic only.
    Phase 4:   feeds into Lagrangian multiplier updates.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._episode_costs: list[dict] = []
        self._ep_sla:  list[float] = []
        self._ep_fair: list[float] = []
        self._ep_mask_applied: list[float] = []

    def _on_step(self) -> bool:
        for info, done in zip(
            self.locals.get("infos", []),
            self.locals.get("dones", []),
        ):
            rb = info.get("reward_breakdown", {})
            self._ep_sla.append( abs(float(rb.get("sla_penalty",      0.0))))
            self._ep_fair.append(abs(float(rb.get("fairness_penalty",  0.0))))
            self._ep_mask_applied.append(float(bool(info.get("action_mask_applied", False))))
            if done:
                mean_sla  = float(np.mean(self._ep_sla))  if self._ep_sla  else 0.0
                mean_fair = float(np.mean(self._ep_fair)) if self._ep_fair else 0.0
                mask_rate = float(np.mean(self._ep_mask_applied)) if self._ep_mask_applied else 0.0
                self._episode_costs.append({"sla": mean_sla, "fairness": mean_fair})
                self.logger.record("costs/episode_mean_sla_penalty",     mean_sla)
                self.logger.record("costs/episode_mean_fairness_penalty", mean_fair)
                self.logger.record("costs/sla_threshold_violated",       float(mean_sla  > THRESHOLD_SLA))
                self.logger.record("costs/fairness_threshold_violated",  float(mean_fair > THRESHOLD_FAIRNESS))
                self.logger.record("costs/episode_action_mask_applied_rate", mask_rate)
                self._ep_sla.clear()
                self._ep_fair.clear()
                self._ep_mask_applied.clear()
        return True

    def _on_training_end(self) -> None:
        if not self._episode_costs:
            return
        all_sla  = [c["sla"]      for c in self._episode_costs]
        all_fair = [c["fairness"] for c in self._episode_costs]
        print(
            f"\n[CostMonitor] mean SLA penalty: {np.mean(all_sla):.4f} "
            f"(threshold={THRESHOLD_SLA}), "
            f"mean fairness penalty: {np.mean(all_fair):.4f} "
            f"(threshold={THRESHOLD_FAIRNESS})"
        )


class RecurrentEvalCallback(BaseCallback):
    """
    Periodic evaluation callback for RecurrentPPO.

    We evaluate with deterministic inference and enforce action masks at
    inference time before env.step().
    """

    def __init__(
        self,
        eval_env: GovWorkflowGymEnv,
        eval_freq: int = 2048,
        n_eval_episodes: int = 3,
        best_model_save_path: str = "results/best_model",
        log_path: str = "results/eval_logs",
        task_id: str = "mixed_urgency_medium",
        verbose: int = 1,
    ):
        super().__init__(verbose=verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_model_save_path = best_model_save_path
        self.log_path = log_path
        self.task_id = task_id
        self._best_grader_score = -np.inf
        os.makedirs(best_model_save_path, exist_ok=True)
        os.makedirs(log_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.eval_freq <= 0 or self.n_calls % self.eval_freq != 0:
            return True

        mean_reward, grader_score = self._run_eval()
        self.logger.record("eval/mean_reward", mean_reward)
        self.logger.record("eval/grader_score", grader_score)

        if grader_score > self._best_grader_score:
            self._best_grader_score = grader_score
            save_path = os.path.join(
                self.best_model_save_path, f"best_grader_recurrent_{self.task_id}"
            )
            self.model.save(save_path)
            if self.verbose:
                print(f"[Eval] New best recurrent grader score: {grader_score:.4f} -> {save_path}")
        return True

    def _run_eval(self) -> tuple[float, float]:
        from app.graders import grade_episode
        from app.tasks import TASKS

        task_cfg = TASKS.get(self.task_id)
        if task_cfg is None:
            return 0.0, 0.0

        rewards: list[float] = []
        scores: list[float] = []

        for ep in range(self.n_eval_episodes):
            env = GovWorkflowGymEnv(self.task_id, seed=task_cfg.seed + ep, hard_action_mask=True)
            obs, _ = env.reset()
            done = False
            ep_reward = 0.0
            lstm_state: Any = None
            episode_start = np.array([True], dtype=bool)

            while not done:
                action, lstm_state = self.model.predict(
                    obs,
                    state=lstm_state,
                    episode_start=episode_start,
                    deterministic=True,
                )
                action_idx = int(np.asarray(action).item())
                masks = env.action_masks()
                if action_idx < 0 or action_idx >= masks.shape[0] or not bool(masks[action_idx]):
                    if masks.shape[0] > 18 and bool(masks[18]):
                        action_idx = 18
                    else:
                        valid = np.flatnonzero(masks)
                        if valid.size > 0:
                            action_idx = int(valid[0])

                obs, reward, terminated, truncated, _ = env.step(action_idx)
                ep_reward += float(reward)
                done = bool(terminated or truncated)
                episode_start = np.array([done], dtype=bool)

            result = grade_episode(env._core_env.state())
            rewards.append(ep_reward)
            scores.append(float(result.score))

        return float(np.mean(rewards)), float(np.mean(scores))
