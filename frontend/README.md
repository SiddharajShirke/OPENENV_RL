# frontend/

Frontend ownership and structure.

- Source-managed React frontend lives in [frontend/react](C:/Users/siddh/OPENENV_RL/frontend/react).
- Built with Vite and served by FastAPI at `/ui`.
- UI is now module-based:
  - `Overview`
  - `Simulation Lab`
  - `Training Studio`
  - `Model Comparison`
- Backend APIs remain under `/api/*`.

Local frontend dev:

1. Start backend:
   - `.\.venv313\Scripts\python.exe scripts\run_local.py --host 0.0.0.0 --port 7860`
2. Start Vite dev server:
   - `cd frontend/react`
   - `npm install`
   - `npm run dev`
3. Open:
   - `http://localhost:5173`

Build for backend serving:

- `cd frontend/react`
- `npm run build`

Deployment path:

- UI route: `/ui`
- Asset route: `/ui/assets/*`
