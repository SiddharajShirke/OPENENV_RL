# ── Path bootstrap ─────────────────────────────────────────────────────────────
from __future__ import annotations
from pathlib import Path

# Load .env file if it exists — must happen before Pydantic Settings reads env vars
from dotenv import load_dotenv
_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_ENV_FILE, override=False)
# override=False means real environment variables always win over .env values
# ──────────────────────────────────────────────────────────────────────────────

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServerSettings(BaseSettings):
    """
    HTTP-server configuration.
    Read from environment variables prefixed SERVER_.
    Example: SERVER_PORT=8080  SERVER_LOG_LEVEL=debug

    Intentionally isolated from EnvSettings — changing server bind
    options never affects simulation behaviour, and vice-versa.
    Both classes are instantiated once at import and treated as
    read-only singletons for the lifetime of the process.
    """

    host: str = Field("0.0.0.0", description="Bind host")
    port: int = Field(7860, description="Bind port — HF Spaces default is 7860")
    log_level: str = Field(
        "info", description="Uvicorn log level: debug | info | warning | error"
    )
    cors_origins: list[str] = Field(
        default=["*"],
        description="Allowed CORS origins. '*' is required for HF Spaces embedding.",
    )
    # NOTE: Keep at 1 when using the in-memory session store.
    # Multiple workers do NOT share process memory.
    # Use Redis + a shared store before increasing workers in production.
    workers: int = Field(
        1, description="Uvicorn worker count — keep at 1 for in-memory sessions"
    )

    model_config = SettingsConfigDict(env_prefix="SERVER_", extra="ignore")


class EnvSettings(BaseSettings):
    """
    Simulation-environment defaults.
    Read from environment variables prefixed ENV_.
    Example: ENV_DEFAULT_TASK_ID=mixed_urgency_medium  ENV_MAX_SESSIONS=50

    Controls the environment kernel only. No effect on network
    binding, logging, or CORS — those belong to ServerSettings.
    """

    default_task_id: str = Field(
        "district_backlog_easy",
        description="Task used when POST /reset is called without an explicit task_id",
    )
    default_seed: int = Field(
        11,
        description="Seed used when POST /reset is called without an explicit seed",
    )
    max_steps_per_episode: int = Field(
        500,
        description="Hard cap on step() calls per session before episode is truncated",
    )
    max_sessions: int = Field(
        100,
        description="Maximum concurrent in-memory sessions. Oldest is evicted when exceeded.",
    )

    model_config = SettingsConfigDict(env_prefix="ENV_", extra="ignore")


# ── Singletons ────────────────────────────────────────────────────────────────
# Loaded exactly once at import time. Never mutated at runtime.
# Tests may monkeypatch individual fields after import if needed.
server_settings = ServerSettings()
env_settings = EnvSettings()