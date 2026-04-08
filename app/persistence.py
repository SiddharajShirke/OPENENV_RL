from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4


def _now() -> float:
    return time.time()


def _as_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=True)


def _from_json(payload: str) -> dict[str, Any]:
    data = json.loads(payload)
    return data if isinstance(data, dict) else {}


def _resolve_data_dir(repo_root: Path) -> Path:
    configured = os.getenv("OPENENV_DATA_DIR") or os.getenv("STORAGE_DATA_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    if Path("/data").exists():
        return Path("/data/openenv_rl").resolve()
    return (repo_root / "outputs" / "persist").resolve()


def _storage_enabled() -> bool:
    raw = str(os.getenv("STORAGE_ENABLED", "true")).strip().lower()
    return raw not in {"0", "false", "no", "off"}


class PersistenceStore:
    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root.resolve()
        self.enabled = _storage_enabled()
        self.data_dir = _resolve_data_dir(self.repo_root)
        self.db_path = self.data_dir / "openenv_state.sqlite3"
        self.training_runs_dir = self.data_dir / "training_runs"
        self._lock = Lock()

        if not self.enabled:
            return

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.training_runs_dir.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS training_jobs (
                    job_id TEXT PRIMARY KEY,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    payload_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS simulation_runs (
                    run_id TEXT PRIMARY KEY,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    task_id TEXT,
                    agent_mode TEXT,
                    status TEXT,
                    payload_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS comparison_runs (
                    comparison_id TEXT PRIMARY KEY,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    task_id TEXT,
                    payload_json TEXT NOT NULL
                );
                """
            )
            conn.commit()

    # Training jobs ---------------------------------------------------------
    def upsert_training_job(self, snapshot: dict[str, Any]) -> None:
        if not self.enabled:
            return
        job_id = str(snapshot.get("job_id") or "")
        if not job_id:
            return
        created_at = float(snapshot.get("created_at") or _now())
        updated_at = float(snapshot.get("updated_at") or _now())
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO training_jobs (job_id, created_at, updated_at, payload_json)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    updated_at = excluded.updated_at,
                    payload_json = excluded.payload_json
                """,
                (job_id, created_at, updated_at, _as_json(snapshot)),
            )
            conn.commit()

    def list_training_jobs(self, limit: int = 500) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        rows: list[dict[str, Any]] = []
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                """
                SELECT payload_json FROM training_jobs
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (max(1, int(limit)),),
            )
            for row in cur.fetchall():
                rows.append(_from_json(str(row["payload_json"])))
        return rows

    # Simulation runs -------------------------------------------------------
    def upsert_simulation_run(
        self,
        *,
        run_id: str,
        task_id: str,
        agent_mode: str,
        status: str,
        payload: dict[str, Any],
    ) -> None:
        if not self.enabled:
            return
        now = _now()
        created_at = float(payload.get("created_at") or now)
        payload = dict(payload)
        payload["run_id"] = run_id
        payload["created_at"] = created_at
        payload["updated_at"] = now
        payload["task_id"] = task_id
        payload["agent_mode"] = agent_mode
        payload["status"] = status
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO simulation_runs (run_id, created_at, updated_at, task_id, agent_mode, status, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    updated_at = excluded.updated_at,
                    task_id = excluded.task_id,
                    agent_mode = excluded.agent_mode,
                    status = excluded.status,
                    payload_json = excluded.payload_json
                """,
                (
                    run_id,
                    created_at,
                    now,
                    task_id,
                    agent_mode,
                    status,
                    _as_json(payload),
                ),
            )
            conn.commit()

    def list_simulation_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        out: list[dict[str, Any]] = []
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                """
                SELECT payload_json FROM simulation_runs
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (max(1, int(limit)),),
            )
            for row in cur.fetchall():
                data = _from_json(str(row["payload_json"]))
                if isinstance(data.get("trace"), list):
                    data["trace_len"] = len(data["trace"])
                    data["has_trace"] = bool(data["trace"])
                    data.pop("trace", None)
                out.append(data)
        return out

    def get_simulation_run(self, run_id: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "SELECT payload_json FROM simulation_runs WHERE run_id = ?",
                (run_id,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return _from_json(str(row["payload_json"]))

    # Comparison runs -------------------------------------------------------
    def create_comparison_run(self, payload: dict[str, Any]) -> str | None:
        if not self.enabled:
            return None
        comparison_id = str(payload.get("comparison_id") or uuid4())
        now = _now()
        body = dict(payload)
        body["comparison_id"] = comparison_id
        body["created_at"] = float(body.get("created_at") or now)
        body["updated_at"] = now
        task_id = str(body.get("task_id") or "")
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO comparison_runs (comparison_id, created_at, updated_at, task_id, payload_json)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(comparison_id) DO UPDATE SET
                    updated_at = excluded.updated_at,
                    task_id = excluded.task_id,
                    payload_json = excluded.payload_json
                """,
                (
                    comparison_id,
                    float(body["created_at"]),
                    now,
                    task_id,
                    _as_json(body),
                ),
            )
            conn.commit()
        return comparison_id

    def list_comparison_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        out: list[dict[str, Any]] = []
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                """
                SELECT payload_json FROM comparison_runs
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (max(1, int(limit)),),
            )
            for row in cur.fetchall():
                out.append(_from_json(str(row["payload_json"])))
        return out

    def get_comparison_run(self, comparison_id: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "SELECT payload_json FROM comparison_runs WHERE comparison_id = ?",
                (comparison_id,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return _from_json(str(row["payload_json"]))
