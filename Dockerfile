# Gov Workflow OpenEnv
# Multi-stage image:
# 1) build Vite frontend assets
# 2) run FastAPI backend and serve built UI under /ui

FROM node:20-slim AS frontend-build
WORKDIR /web

COPY frontend/react/package.json frontend/react/package-lock.json ./frontend/react/
RUN cd frontend/react && npm ci --no-audit --no-fund

COPY frontend/react ./frontend/react
RUN cd frontend/react && npm run build


FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OPENENV_DATA_DIR=/data/openenv_rl \
    STORAGE_ENABLED=true \
    PORT=7860

WORKDIR /app

# Runtime OS dependencies (torch/sb3 commonly require libgomp at runtime)
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements_rl.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt \
    && python -m pip install -r requirements_rl.txt

COPY . .
COPY --from=frontend-build /web/frontend/react/dist ./app/web/vite_dist

RUN mkdir -p /data/openenv_rl \
    && useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app /data/openenv_rl

USER appuser

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/health', timeout=3)" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
