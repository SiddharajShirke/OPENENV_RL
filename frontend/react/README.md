# react/

Vite + React frontend for the Gov Workflow OpenEnv console.

Commands:

- `npm install`
- `npm run dev` (local dev on `http://localhost:5173`, proxies `/api` to `http://localhost:7860`)
- `npm run build` (production build for Docker/HF)
- `npm run preview`

If you see `ERR_CONNECTION_REFUSED` on `/api/*`:

- Start backend first on port `7860`
- Or set a custom dev proxy target:
  - PowerShell: `$env:VITE_DEV_API_TARGET='http://127.0.0.1:7860'`
  - Then run `npm run dev`

Modules:

- `Overview`: project and environment summary
- `Simulation Lab`: dynamic real-world workflow simulation (baseline / inference-like / trained RL)
- `Training Studio`: launch and monitor background RL training jobs
- `Model Comparison`: baseline vs trained model score comparison on the same task
