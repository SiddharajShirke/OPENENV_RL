#!/usr/bin/env python3
"""
Local OpenEnv validation helper.

Checks:
1. openenv.yaml exists and contains required sections
2. environment/model import paths are importable
3. optional: `openenv validate` when CLI is installed
"""

from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


REQUIRED_TOP_LEVEL = ("name", "entrypoint", "environment", "tasks", "api")


def _import_path(path: str) -> Any:
    module_name, _, obj_name = path.rpartition(".")
    if not module_name or not obj_name:
        raise ValueError(f"Invalid import path: {path!r}")
    module = importlib.import_module(module_name)
    return getattr(module, obj_name)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate OpenEnv environment shape")
    parser.add_argument("--repo", default=".")
    parser.add_argument(
        "--skip-openenv-cli",
        action="store_true",
        help="Skip invoking `openenv validate`",
    )
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    cfg_path = repo / "openenv.yaml"
    if not cfg_path.exists():
        print(f"[FAIL] Missing {cfg_path}")
        return 1

    config = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if config.get("spec_version") != 1:
        print("[FAIL] openenv.yaml must declare spec_version: 1")
        return 1

    for key in REQUIRED_TOP_LEVEL:
        if key not in config:
            print(f"[FAIL] openenv.yaml missing required top-level key: {key}")
            return 1

    env_cfg = config["environment"]
    entrypoint = config["entrypoint"]

    for field in ("module", "object"):
        if field not in entrypoint:
            print(f"[FAIL] entrypoint missing field: {field}")
            return 1

    _import_path(f"{entrypoint['module']}.{entrypoint['object']}")
    _import_path(env_cfg["class"])
    _import_path(env_cfg["observation_model"])
    _import_path(env_cfg["action_model"])
    _import_path(env_cfg["reward_model"])
    _import_path(env_cfg["state_model"])
    _import_path(env_cfg["step_info_model"])
    print("[OK] openenv.yaml imports are valid")

    tasks = config.get("tasks", [])
    if len(tasks) < 3:
        print("[FAIL] Need at least 3 tasks in openenv.yaml")
        return 1
    print(f"[OK] task count={len(tasks)}")

    if not args.skip_openenv_cli:
        try:
            proc = subprocess.run(
                ["openenv", "validate"],
                cwd=str(repo),
                check=False,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError:
            print("[WARN] `openenv` CLI not found; skipping `openenv validate`")
        else:
            if proc.returncode != 0:
                print("[FAIL] `openenv validate` failed")
                if proc.stdout:
                    print(proc.stdout.rstrip())
                if proc.stderr:
                    print(proc.stderr.rstrip())
                return proc.returncode
            print("[OK] `openenv validate` passed")

    print("[DONE] validation complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
