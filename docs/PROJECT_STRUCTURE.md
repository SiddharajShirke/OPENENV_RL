# Project Structure (Judge-Friendly)

This repository keeps runtime-critical files in their original paths for deployment safety.
No existing files were deleted.

## Top-Level Layout

- `app/` - core environment logic and FastAPI server
- `app/web/` - deployed React frontend assets served by backend at `/ui`
- `frontend/` - frontend ownership docs and reserved source folder for future split components
- `rl/` - reinforcement-learning wrappers, training, evaluation, configs
- `tests/` - deterministic unit/integration test suites
- `scripts/` - operational scripts (local run, validation, benchmark ladder)
- `docs/` - judge-facing documentation and phase notes
- `openenv.yaml` - OpenEnv manifest
- `inference.py` - OpenEnv inference entrypoint
- `baseline_openai.py` - CLI baseline workflow
- `Dockerfile` - deployment image

## Deployment-Critical Paths

- API app import path: `app.main:app`
- Frontend route: `/ui` (served from `app/web/index.html`)
- RL training entrypoint: `python -m rl.train_ppo`
- RL evaluation entrypoint: `python -m rl.evaluate`
- OpenEnv config: `openenv.yaml`

## Phase Mapping

- Phase 1: `rl/feature_builder.py`, `rl/action_mask.py`, `rl/gym_wrapper.py`, `rl/train_ppo.py`
- Phase 2: `rl/curriculum.py`, `rl/configs/curriculum.yaml`, `tests/test_curriculum.py`
- Phase 3: `rl/train_recurrent.py`, `rl/configs/recurrent.yaml`, `tests/test_rl_evaluate.py`
- Phase 3+: reserved in existing `rl/` module structure

## Judge Quick Navigation

1. Environment behavior: `app/env.py`, `app/reward.py`, `app/graders.py`
2. OpenEnv compliance + inference: `openenv.yaml`, `inference.py`
3. Frontend behavior: `app/web/react_app.js`, `docs/FRONTEND_WORKFLOW.md`
4. RL implementation: `rl/`
5. Validation: `tests/`, `scripts/validate_env.py`, `scripts/validate-submission.sh`