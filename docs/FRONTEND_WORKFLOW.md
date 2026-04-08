# Frontend Workflow

The frontend is React-based, backend-driven, and served directly by FastAPI.

## Access

- UI: `/ui`
- Assets: `/ui/assets/*`
- API namespace: `/api/*`

## What Is Visible in UI

1. OpenEnv API execution (`reset` / `step` / `state` / `grade`)
2. Heuristic baseline agent runs (`/api/autostep`, `/api/benchmark`)
3. Trained RL model execution (Phase 2/3 checkpoints via `/api/rl/run`)
4. Trained RL evaluation across tasks (`/api/rl/evaluate`)
5. Script-level workflow visibility for:
   - `baseline_openai.py`
   - `inference.py`

## Frontend API Surface

- Core:
  - `GET /api/health`
  - `GET /api/tasks`
  - `GET /api/agents`
  - `POST /api/reset`
  - `POST /api/step`
  - `POST /api/state`
  - `POST /api/grade`
  - `GET /api/sessions`
  - `DELETE /api/sessions/{session_id}`
- Baseline execution:
  - `POST /api/autostep`
  - `POST /api/benchmark`
- Workflow visibility:
  - `GET /api/workflows/components`
  - `POST /api/workflows/run`
- RL visibility/execution:
  - `GET /api/rl/models`
  - `POST /api/rl/run`
  - `POST /api/rl/evaluate`

## Deployment Notes

- No Node.js build is required for serving the current frontend.
- Backend startup remains `app.main:app`.
- Frontend does not call external LLM providers directly.
