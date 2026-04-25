# rl/

Reinforcement learning module.

- `gov_workflow_env.py`: Gymnasium adapter around `app.env.GovWorkflowEnv`
- `feature_builder.py`: `ObservationModel` -> 84-dim float32 vector (`OBS_DIM=84`)
- `action_mask.py`: structural action masks (`N_ACTIONS=28`)
- `curriculum.py`: staged task scheduler (Phase 2/3)
- `train_ppo.py`: Phase 1 and Phase 2 training entrypoint
- `train_recurrent.py`: Phase 3 recurrent PPO entrypoint
- `evaluate.py`: deterministic evaluation on grader metrics (`--task` / `--tasks`)
- `eval_grader.py`: task-level grader evaluation helper with optional plots
- `plot_training.py`: training-curve report helper from monitor/TensorBoard artifacts
- `callbacks.py`: eval and cost-monitor callbacks
- `cost_tracker.py`: episode-level reward/cost extraction helpers
- `configs/`: YAML configs for PPO/recurrent training
  - `ppo_easy.yaml`: standard Phase 1 config
  - `ppo_easy_aggressive.yaml`: aggressive Phase 1 tuning profile for plateau recovery

## CLI Compatibility Notes

- Training scripts accept both `--n-envs` and `--n_envs`.
- `train_ppo.py` accepts `--task` as a compatibility alias:
  - Phase 1 only supports `district_backlog_easy`
  - Phase 2 ignores `--task` and uses curriculum sampling
- `train_ppo.py` supports `--resume <checkpoint>` for Phase 1 continuation runs.
- `train_recurrent.py` accepts `--task` to override recurrent eval callback task.

## Artifact Paths

- Training/eval outputs are written under `results/`:
  - `results/best_model/*`
  - `results/runs/*`
  - `results/eval_logs/*`
