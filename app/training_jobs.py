from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import threading
import time
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from app.persistence import PersistenceStore

Status = Literal["queued", "running", "completed", "failed", "stopped"]

_PROGRESS_RE = re.compile(r"(\d[\d,]*)/(\d[\d,]*)")
_METRIC_ROW_RE = re.compile(r"\|\s*([a-zA-Z0-9_ ]+?)\s*\|\s*(-?\d+(?:\.\d+)?)\s*\|")
_EVAL_PROGRESS_RE = re.compile(
    r"Eval\s+num_timesteps=(\d+),\s*episode_reward=([-]?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_EVAL_ROW_RE = re.compile(
    r"^\[Eval\]\s+([a-z_]+)\s+score=([0-9.]+)\s+reward=([-0-9.]+)\s+completed=(\d+)\s+sla_breaches=(\d+)$"
)
_AVG_RE = re.compile(r"^\[Eval\]\s+Average grader score:\s+([0-9.]+)$")
_BEST_GRADER_RE = re.compile(
    r"\[Eval\]\s+New best(?: recurrent)? grader score:\s+([0-9.]+)",
    re.IGNORECASE,
)


def _now() -> float:
    return time.time()


def _tail_append(lines: list[str], line: str, max_size: int = 500) -> None:
    lines.append(line.rstrip("\n"))
    if len(lines) > max_size:
        del lines[: len(lines) - max_size]


def _normalize_metric_key(raw: str) -> str:
    return raw.strip().lower().replace(" ", "_")


def _parse_eval(stdout: str) -> tuple[list[dict[str, Any]], float | None]:
    rows: list[dict[str, Any]] = []
    avg: float | None = None
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        row = _EVAL_ROW_RE.match(line)
        if row:
            rows.append(
                {
                    "task_id": row.group(1),
                    "grader_score": float(row.group(2)),
                    "total_reward": float(row.group(3)),
                    "total_completed": int(row.group(4)),
                    "total_sla_breaches": int(row.group(5)),
                }
            )
            continue
        m = _AVG_RE.match(line)
        if m:
            avg = float(m.group(1))
    return rows, avg


@dataclass
class TrainingJob:
    job_id: str
    phase: int
    timesteps: int
    n_envs: int
    seed: int
    config_path: str
    created_at: float = field(default_factory=_now)
    started_at: float | None = None
    updated_at: float = field(default_factory=_now)
    ended_at: float | None = None
    status: Status = "queued"
    progress: float = 0.0
    process_id: int | None = None
    command: list[str] = field(default_factory=list)
    output_model_path: str | None = None
    output_model_name: str | None = None
    latest_metrics: dict[str, float] = field(default_factory=dict)
    metric_history: list[dict[str, Any]] = field(default_factory=list)
    evaluation_rows: list[dict[str, Any]] = field(default_factory=list)
    evaluation_avg_score: float | None = None
    logs_tail: list[str] = field(default_factory=list)
    error_message: str | None = None
    return_code: int | None = None

    process: subprocess.Popen[str] | None = field(default=None, repr=False)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    last_persist_at: float = field(default_factory=lambda: 0.0, repr=False)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "job_id": self.job_id,
                "phase": self.phase,
                "timesteps": self.timesteps,
                "n_envs": self.n_envs,
                "seed": self.seed,
                "config_path": self.config_path,
                "created_at": self.created_at,
                "started_at": self.started_at,
                "updated_at": self.updated_at,
                "ended_at": self.ended_at,
                "status": self.status,
                "progress": self.progress,
                "process_id": self.process_id,
                "command": self.command,
                "output_model_path": self.output_model_path,
                "output_model_name": self.output_model_name,
                "latest_metrics": dict(self.latest_metrics),
                "metric_history": list(self.metric_history),
                "evaluation_rows": list(self.evaluation_rows),
                "evaluation_avg_score": self.evaluation_avg_score,
                "logs_tail": list(self.logs_tail),
                "error_message": self.error_message,
                "return_code": self.return_code,
            }


class TrainingJobManager:
    def __init__(self, repo_root: Path, persistence: PersistenceStore | None = None) -> None:
        self._repo_root = repo_root
        self._persistence = persistence
        self._jobs: dict[str, TrainingJob] = {}
        self._lock = threading.Lock()
        self._training_runs_root = (
            self._persistence.training_runs_dir
            if self._persistence is not None and self._persistence.enabled
            else self._repo_root / "results" / "training_runs"
        )
        self._load_persisted_jobs()

    def _load_persisted_jobs(self) -> None:
        if self._persistence is None or not self._persistence.enabled:
            return
        persisted = self._persistence.list_training_jobs(limit=500)
        with self._lock:
            for snap in persisted:
                try:
                    job = TrainingJob(
                        job_id=str(snap["job_id"]),
                        phase=int(snap["phase"]),
                        timesteps=int(snap["timesteps"]),
                        n_envs=int(snap["n_envs"]),
                        seed=int(snap["seed"]),
                        config_path=str(snap.get("config_path") or ""),
                        created_at=float(snap.get("created_at") or _now()),
                        started_at=float(snap["started_at"]) if snap.get("started_at") is not None else None,
                        updated_at=float(snap.get("updated_at") or _now()),
                        ended_at=float(snap["ended_at"]) if snap.get("ended_at") is not None else None,
                        status=str(snap.get("status") or "failed"),
                        progress=float(snap.get("progress") or 0.0),
                        process_id=int(snap["process_id"]) if snap.get("process_id") is not None else None,
                        command=list(snap.get("command") or []),
                        output_model_path=snap.get("output_model_path"),
                        output_model_name=snap.get("output_model_name"),
                        latest_metrics=dict(snap.get("latest_metrics") or {}),
                        metric_history=list(snap.get("metric_history") or []),
                        evaluation_rows=list(snap.get("evaluation_rows") or []),
                        evaluation_avg_score=(
                            float(snap["evaluation_avg_score"])
                            if snap.get("evaluation_avg_score") is not None
                            else None
                        ),
                        logs_tail=list(snap.get("logs_tail") or []),
                        error_message=snap.get("error_message"),
                        return_code=int(snap["return_code"]) if snap.get("return_code") is not None else None,
                    )
                except Exception:
                    continue

                # Process handles cannot survive a server restart. Recover to terminal state.
                if job.status in ("queued", "running"):
                    job.status = "failed"
                    msg = "Recovered after restart: previous process state unavailable."
                    job.error_message = f"{job.error_message} {msg}".strip() if job.error_message else msg
                    if job.ended_at is None:
                        job.ended_at = _now()
                job.process = None
                self._jobs[job.job_id] = job

    def clear_jobs(self, *, clear_artifacts: bool = False) -> int:
        to_stop: list[subprocess.Popen[str]] = []
        with self._lock:
            removed = len(self._jobs)
            for job in self._jobs.values():
                with job.lock:
                    proc = job.process
                    if proc is not None and job.status in ("queued", "running"):
                        to_stop.append(proc)
            self._jobs.clear()
        for proc in to_stop:
            try:
                proc.terminate()
            except Exception:
                pass
        if self._persistence is not None and self._persistence.enabled:
            self._persistence.clear_training_jobs()
        if clear_artifacts:
            try:
                if self._training_runs_root.exists():
                    shutil.rmtree(self._training_runs_root, ignore_errors=True)
                self._training_runs_root.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
        return removed

    def _persist_job(self, job: TrainingJob) -> None:
        if self._persistence is None or not self._persistence.enabled:
            return
        snapshot = job.snapshot()
        self._persistence.upsert_training_job(snapshot)
        with job.lock:
            job.last_persist_at = _now()

    def list_jobs(self) -> list[dict[str, Any]]:
        with self._lock:
            jobs = list(self._jobs.values())
        jobs.sort(key=lambda x: x.created_at, reverse=True)
        return [job.snapshot() for job in jobs]

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
        return None if job is None else job.snapshot()

    def start_job(
        self,
        *,
        phase: int,
        timesteps: int,
        n_envs: int,
        seed: int | None,
        config_path: str | None,
    ) -> dict[str, Any]:
        job_id = str(uuid4())
        job_seed = int(seed if seed is not None else int(time.time()) % 1_000_000)
        cfg = config_path or (
            "rl/configs/ppo_easy.yaml" if phase == 1 else "rl/configs/curriculum.yaml"
        )
        job = TrainingJob(
            job_id=job_id,
            phase=phase,
            timesteps=timesteps,
            n_envs=n_envs,
            seed=job_seed,
            config_path=cfg,
        )

        with self._lock:
            self._jobs[job_id] = job

        cmd = [
            sys.executable,
            "-u",
            "-m",
            "rl.train_ppo",
            "--phase",
            str(phase),
            "--timesteps",
            str(timesteps),
            "--n-envs",
            str(n_envs),
            "--seed",
            str(job_seed),
        ]
        if phase == 1:
            # Keep Phase 1 UI responsive by emitting multiple eval checkpoints
            # across the requested run length instead of only near the end.
            phase1_eval_freq = max(128, int((timesteps / max(n_envs, 1)) / 15))
            cmd.extend(
                [
                    "--phase1-config",
                    cfg,
                    "--phase1-eval-freq",
                    str(phase1_eval_freq),
                ]
            )
        else:
            cmd.extend(["--phase2-config", cfg])

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        proc = subprocess.Popen(
            cmd,
            cwd=str(self._repo_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        with job.lock:
            job.command = cmd
            job.status = "running"
            job.started_at = _now()
            job.updated_at = _now()
            job.process_id = proc.pid
            job.process = proc
            _tail_append(job.logs_tail, f"[training_jobs] started pid={proc.pid}")
            _tail_append(job.logs_tail, f"[training_jobs] command: {' '.join(cmd)}")
        self._persist_job(job)

        t = threading.Thread(target=self._watch_job, args=(job,), daemon=True)
        t.start()

        return job.snapshot()

    @staticmethod
    def _append_metric_point_locked(
        job: TrainingJob,
        *,
        timesteps: float | None,
        reward: float | None = None,
        score: float | None = None,
        source: str | None = None,
        max_points: int = 5000,
    ) -> None:
        """
        Append (or merge) a structured metric point while holding job.lock.
        """
        if timesteps is None or not math.isfinite(float(timesteps)):
            return

        payload: dict[str, Any] = {"t": float(timesteps)}
        if reward is not None and math.isfinite(float(reward)):
            payload["ep_rew_mean"] = float(reward)
        if score is not None and math.isfinite(float(score)):
            payload["grader_score"] = float(score)
        if source:
            payload["source"] = str(source)

        if "ep_rew_mean" not in payload and "grader_score" not in payload:
            return

        if job.metric_history and float(job.metric_history[-1].get("t", -1.0)) == float(payload["t"]):
            job.metric_history[-1].update(payload)
        else:
            job.metric_history.append(payload)

        if len(job.metric_history) > max_points:
            del job.metric_history[: len(job.metric_history) - max_points]

    def stop_job(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            return None

        with job.lock:
            proc = job.process
            if proc is None or job.status not in ("running", "queued"):
                return job.snapshot()
            job.status = "stopped"
            job.updated_at = _now()
        self._persist_job(job)

        try:
            proc.terminate()
        except Exception:
            pass
        return job.snapshot()

    def delete_job(self, job_id: str, *, clear_artifacts: bool = False) -> bool:
        with self._lock:
            job = self._jobs.pop(job_id, None)
        if job is None:
            return False

        with job.lock:
            proc = job.process
            status = job.status
            output_model_path = job.output_model_path

        if proc is not None and status in ("queued", "running"):
            try:
                proc.terminate()
            except Exception:
                pass

        if self._persistence is not None and self._persistence.enabled:
            self._persistence.delete_training_job(job_id)

        if clear_artifacts and output_model_path:
            try:
                out = Path(output_model_path)
                if out.exists() and out.is_file():
                    out.unlink(missing_ok=True)
                parent = out.parent
                if parent.exists() and parent.is_dir() and not any(parent.iterdir()):
                    parent.rmdir()
            except Exception:
                pass
        return True

    def _watch_job(self, job: TrainingJob) -> None:
        proc = job.process
        if proc is None or proc.stdout is None:
            with job.lock:
                job.status = "failed"
                job.error_message = "Training process failed to start."
                job.updated_at = _now()
                job.ended_at = _now()
            self._persist_job(job)
            return

        for line in proc.stdout:
            self._update_from_line(job, line)

        return_code = proc.wait()
        with job.lock:
            job.return_code = int(return_code)
            if job.status == "stopped":
                job.ended_at = _now()
                job.updated_at = _now()
                job.process = None
                return
            if return_code == 0:
                job.status = "completed"
                job.progress = 1.0
            else:
                job.status = "failed"
                base_error = f"Training exited with code {return_code}."
                if not job.logs_tail:
                    _tail_append(
                        job.logs_tail,
                        "[training_jobs] Process ended before producing logs. "
                        "Check RL dependencies/environment and training command arguments.",
                    )
                job.error_message = base_error
            job.ended_at = _now()
            job.updated_at = _now()
            job.process = None
        self._persist_job(job)

        if return_code == 0:
            self._finalize_artifacts(job)

    def _update_from_line(self, job: TrainingJob, line: str) -> None:
        line = line.rstrip("\n")
        should_persist = False
        with job.lock:
            _tail_append(job.logs_tail, line)
            job.updated_at = _now()

            p = _PROGRESS_RE.search(line)
            if p:
                num = int(p.group(1).replace(",", ""))
                den = int(p.group(2).replace(",", ""))
                if den > 0:
                    job.progress = max(0.0, min(1.0, num / den))

            ep = _EVAL_PROGRESS_RE.search(line)
            if ep:
                ts = int(ep.group(1))
                rew = float(ep.group(2))
                job.latest_metrics["total_timesteps"] = float(ts)
                job.latest_metrics["ep_rew_mean"] = rew
                self._append_metric_point_locked(
                    job,
                    timesteps=float(ts),
                    reward=rew,
                    source="eval_progress",
                )
                if job.timesteps > 0:
                    job.progress = max(0.0, min(1.0, ts / float(job.timesteps)))

            m = _METRIC_ROW_RE.search(line)
            if m:
                key = _normalize_metric_key(m.group(1))
                val = float(m.group(2))
                interesting = {
                    "total_timesteps",
                    "ep_rew_mean",
                    "ep_len_mean",
                    "grader_score",
                    "mean_reward",
                    "mean_ep_length",
                    "episode_mean_sla_penalty",
                    "episode_mean_fairness_penalty",
                    "explained_variance",
                    "approx_kl",
                }
                if key in interesting:
                    job.latest_metrics[key] = val
                    current_ts = job.latest_metrics.get("total_timesteps")
                    if key == "total_timesteps":
                        self._append_metric_point_locked(
                            job,
                            timesteps=val,
                            reward=job.latest_metrics.get("ep_rew_mean"),
                            score=job.latest_metrics.get("grader_score") or job.latest_metrics.get("avg_grader_score"),
                            source="metrics_row_ts",
                        )
                    elif key in {"ep_rew_mean", "mean_reward"}:
                        self._append_metric_point_locked(
                            job,
                            timesteps=float(current_ts) if current_ts is not None else None,
                            reward=val,
                            source="metrics_row_reward",
                        )
                    elif key in {"grader_score", "avg_grader_score"}:
                        self._append_metric_point_locked(
                            job,
                            timesteps=float(current_ts) if current_ts is not None else None,
                            score=val,
                            source="metrics_row_score",
                        )

            best = _BEST_GRADER_RE.search(line)
            if best:
                score = float(best.group(1))
                job.latest_metrics["grader_score"] = score
                fallback_ts = (
                    float(job.latest_metrics.get("total_timesteps"))
                    if "total_timesteps" in job.latest_metrics
                    else float(job.metric_history[-1]["t"]) if job.metric_history else 0.0
                )
                self._append_metric_point_locked(
                    job,
                    timesteps=fallback_ts if fallback_ts > 0 else float(len(job.metric_history) + 1),
                    score=score,
                    source="best_grader",
                )

            avg_line = _AVG_RE.match(line.strip())
            if avg_line:
                avg_score = float(avg_line.group(1))
                job.latest_metrics["avg_grader_score"] = avg_score
                fallback_ts = (
                    float(job.latest_metrics.get("total_timesteps"))
                    if "total_timesteps" in job.latest_metrics
                    else float(job.metric_history[-1]["t"]) if job.metric_history else 0.0
                )
                self._append_metric_point_locked(
                    job,
                    timesteps=fallback_ts if fallback_ts > 0 else float(len(job.metric_history) + 1),
                    score=avg_score,
                    source="avg_grader",
                )
            if job.updated_at - job.last_persist_at >= 1.5:
                should_persist = True
        if should_persist:
            self._persist_job(job)

    def _finalize_artifacts(self, job: TrainingJob) -> None:
        src_name = "phase1_final.zip" if job.phase == 1 else "phase2_final.zip"
        src = self._repo_root / "results" / "best_model" / src_name
        run_dir = self._training_runs_root / job.job_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Keep a mirror under repo/results for local developer convenience.
        mirror_dir = self._repo_root / "results" / "training_runs" / job.job_id
        if mirror_dir != run_dir:
            mirror_dir.mkdir(parents=True, exist_ok=True)

        if src.exists():
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            unique_name = f"phase{job.phase}_seed{job.seed}_{ts}_{job.job_id[:8]}.zip"
            out = run_dir / unique_name
            shutil.copy2(src, out)
            if mirror_dir != run_dir:
                try:
                    shutil.copy2(src, mirror_dir / unique_name)
                except Exception:
                    pass
            with job.lock:
                job.output_model_path = str(out.resolve())
                job.output_model_name = unique_name
                job.updated_at = _now()

            model_type = "maskable"
            eval_cmd = [
                sys.executable,
                "-m",
                "rl.evaluate",
                "--model",
                str(out),
                "--episodes",
                "3",
                "--model-type",
                model_type,
            ]
            proc = subprocess.run(
                eval_cmd,
                cwd=str(self._repo_root),
                env=os.environ.copy(),
                capture_output=True,
                text=True,
                check=False,
            )
            rows, avg = _parse_eval(proc.stdout or "")
            with job.lock:
                job.evaluation_rows = rows
                job.evaluation_avg_score = avg
                if avg is not None:
                    job.latest_metrics["avg_grader_score"] = float(avg)
                    fallback_ts = (
                        float(job.latest_metrics.get("total_timesteps"))
                        if "total_timesteps" in job.latest_metrics
                        else float(job.timesteps)
                    )
                    self._append_metric_point_locked(
                        job,
                        timesteps=fallback_ts if fallback_ts > 0 else float(len(job.metric_history) + 1),
                        score=float(avg),
                        source="final_eval_avg",
                    )
                _tail_append(job.logs_tail, "----- EVALUATION -----")
                for ln in (proc.stdout or "").splitlines():
                    _tail_append(job.logs_tail, ln)
                if proc.returncode != 0 and not job.error_message:
                    job.error_message = f"Evaluation exited with code {proc.returncode}."
                job.updated_at = _now()
            self._persist_job(job)
        else:
            self._persist_job(job)
