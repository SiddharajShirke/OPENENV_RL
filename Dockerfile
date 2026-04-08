# Dockerfile - Gov Workflow OpenEnv
# Multi-stage build:
# 1) build Vite frontend
# 2) run FastAPI backend and serve built UI at /ui

FROM node:20-slim AS frontend-build
WORKDIR /web

COPY frontend/react/package.json ./frontend/react/package.json
COPY frontend/react/package-lock.json ./frontend/react/package-lock.json
RUN cd frontend/react && npm ci --no-audit --no-fund

COPY frontend/react ./frontend/react
RUN cd frontend/react && npm run build


FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OPENENV_DATA_DIR=/data/openenv_rl \
    STORAGE_ENABLED=true

WORKDIR /app

COPY requirements.txt .
COPY requirements_rl.txt .
RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -r requirements_rl.txt

COPY . .
COPY --from=frontend-build /web/frontend/react/dist ./app/web/vite_dist
RUN mkdir -p /data/openenv_rl

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
