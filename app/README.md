# app/

Core environment and API layer.

- `main.py`: FastAPI app and endpoints
- `env.py`: GovWorkflowEnv simulation kernel
- `models.py`: Pydantic action/observation/reward/state models
- `tasks.py`: easy/medium/hard deterministic task configs
- `graders.py`: deterministic task scoring (0.0 to 1.0)
- `reward.py`: dense reward breakdown
- `baselines.py`: heuristic baseline policies
- `web/`: frontend assets served by FastAPI at `/ui`
  - `vite_dist/`: production Vite build output copied during Docker build
  - legacy files (`index.html`, `react_app.js`, `styles.css`) remain as local fallback

Additional frontend-focused APIs in `main.py`:
- `/api/workflows/components`
- `/api/workflows/run`
- `/api/rl/models`
- `/api/rl/run`
- `/api/rl/evaluate`
- `/api/simulation/run`
- `/api/training/jobs`
