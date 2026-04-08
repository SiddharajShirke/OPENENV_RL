# Phase 2 Implementation Notes

Phase 2 goal: Curriculum PPO across easy, medium, and hard tasks with deterministic evaluation discipline.

## Implemented Components

- `rl/curriculum.py`
  - `CurriculumScheduler` with staged task sampling:
    - Stage 1 (0%-30%): easy only
    - Stage 2 (30%-70%): easy + medium
    - Stage 3 (70%-100%): all 3 tasks with configurable weights
- `rl/configs/curriculum.yaml`
  - curriculum fractions and weights
  - PPO hyperparameters for Phase 2
- `rl/train_ppo.py`
  - `--phase 2` training path wired to curriculum scheduler
  - default config path uses `rl/configs/curriculum.yaml`
  - backward compatibility fallback to `rl/configs/ppo_curriculum.yaml`
  - explicit CLI args: `--phase1-config`, `--phase2-config`
- `tests/test_curriculum.py`
  - stage transitions
  - stage-1 easy-only enforcement
  - stage-3 all-task sampling
  - deterministic task seed invariants

## Operational Notes

- Existing 28-action design is preserved.
- Existing task IDs and grader logic are unchanged.
- No files were deleted as part of structure cleanup.

## Commands (using existing .venv313)

- Train Phase 1:
  - `.\\.venv313\\Scripts\\python.exe -m rl.train_ppo --phase 1 --timesteps 200000 --n-envs 4 --seed 42`
- Train Phase 2:
  - `.\\.venv313\\Scripts\\python.exe -m rl.train_ppo --phase 2 --timesteps 500000 --n-envs 4 --seed 42 --phase2-config rl/configs/curriculum.yaml`
- Train Phase 2 (tuned continuation):
  - `.\\.venv313\\Scripts\\python.exe -m rl.train_ppo --phase 2 --timesteps 300000 --n-envs 4 --seed 42 --phase2-config rl/configs/curriculum_tuned.yaml`
- Evaluate trained model:
  - `.\\.venv313\\Scripts\\python.exe -m rl.evaluate --model results/best_model/phase2_final.zip --episodes 3`
