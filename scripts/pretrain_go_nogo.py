"""
Pre-train checklist + GO/NO-GO gate for Gov Workflow RL Phase 1.

This script validates the local training stack without running training.
Use it before starting Phase 1 retraining.

Usage:
    python scripts/pretrain_go_nogo.py
    python scripts/pretrain_go_nogo.py --run-tests
"""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PHASE1_TASK = "district_backlog_easy"
EXPECTED_OBS_DIM = 84
EXPECTED_ACTIONS = 28


@dataclass
class CheckResult:
    name: str
    status: str  # PASS | WARN | FAIL
    detail: str


def _run_cmd(cmd: list[str], cwd: Path | None = None) -> tuple[int, str, str]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd or ROOT),
        capture_output=True,
        text=True,
    )
    return proc.returncode, proc.stdout, proc.stderr


def check_required_files() -> CheckResult:
    required = [
        "rl/train_ppo.py",
        "rl/train_recurrent.py",
        "rl/gov_workflow_env.py",
        "rl/feature_builder.py",
        "rl/action_mask.py",
        "rl/callbacks.py",
        "rl/curriculum.py",
        "rl/cost_tracker.py",
        "rl/evaluate.py",
        "rl/eval_grader.py",
        "rl/plot_training.py",
        "rl/configs/ppo_easy.yaml",
        "app/env.py",
        "app/models.py",
        "app/tasks.py",
        "app/reward.py",
        "app/graders.py",
    ]
    missing = [p for p in required if not (ROOT / p).exists()]
    if missing:
        return CheckResult(
            name="required_files",
            status="FAIL",
            detail="Missing files: " + ", ".join(missing),
        )
    return CheckResult(
        name="required_files",
        status="PASS",
        detail=f"{len(required)} required files present",
    )


def check_python_imports() -> CheckResult:
    modules = [
        "yaml",
        "numpy",
        "gymnasium",
        "torch",
        "stable_baselines3",
        "sb3_contrib",
        "tensorboard",
        "rl.train_ppo",
        "rl.train_recurrent",
        "rl.gov_workflow_env",
        "rl.feature_builder",
        "rl.action_mask",
        "rl.callbacks",
        "rl.evaluate",
        "rl.eval_grader",
        "app.env",
        "app.tasks",
        "app.graders",
    ]
    failed: list[str] = []
    for mod in modules:
        try:
            importlib.import_module(mod)
        except Exception:
            failed.append(mod)
    if failed:
        return CheckResult(
            name="python_imports",
            status="FAIL",
            detail="Import failures: " + ", ".join(failed),
        )
    return CheckResult(
        name="python_imports",
        status="PASS",
        detail=f"{len(modules)} modules import cleanly",
    )


def check_compile() -> CheckResult:
    targets = [
        "rl/train_ppo.py",
        "rl/train_recurrent.py",
        "rl/gov_workflow_env.py",
        "rl/feature_builder.py",
        "rl/action_mask.py",
        "rl/callbacks.py",
        "rl/evaluate.py",
        "rl/eval_grader.py",
        "app/env.py",
        "app/reward.py",
        "app/graders.py",
        "app/tasks.py",
    ]
    cmd = [sys.executable, "-m", "py_compile", *targets]
    rc, _out, err = _run_cmd(cmd, ROOT)
    if rc != 0:
        return CheckResult(
            name="py_compile",
            status="FAIL",
            detail=err.strip() or "py_compile failed",
        )
    return CheckResult(
        name="py_compile",
        status="PASS",
        detail=f"{len(targets)} files compiled successfully",
    )


def check_env_contract() -> CheckResult:
    try:
        from rl.gov_workflow_env import GovWorkflowGymEnv

        env = GovWorkflowGymEnv(task_id=PHASE1_TASK, seed=42)
        obs, info = env.reset(seed=42)
        masks = env.action_masks()
        _obs2, reward, terminated, truncated, step_info = env.step(18)

        problems: list[str] = []
        if tuple(obs.shape) != (EXPECTED_OBS_DIM,):
            problems.append(f"obs shape={tuple(obs.shape)} expected={(EXPECTED_OBS_DIM,)}")
        if int(env.action_space.n) != EXPECTED_ACTIONS:
            problems.append(f"action_space={env.action_space.n} expected={EXPECTED_ACTIONS}")
        if len(masks) != EXPECTED_ACTIONS:
            problems.append(f"mask_len={len(masks)} expected={EXPECTED_ACTIONS}")
        if int(sum(bool(x) for x in masks)) <= 0:
            problems.append("all actions masked")
        if not isinstance(info, dict):
            problems.append("reset info is not dict")
        if not isinstance(step_info, dict):
            problems.append("step info is not dict")
        if not isinstance(float(reward), float):
            problems.append("reward not float-castable")
        if not isinstance(bool(terminated), bool) or not isinstance(bool(truncated), bool):
            problems.append("terminated/truncated invalid type")

        if problems:
            return CheckResult(
                name="gym_env_contract",
                status="FAIL",
                detail="; ".join(problems),
            )
        return CheckResult(
            name="gym_env_contract",
            status="PASS",
            detail=f"obs={obs.shape}, action_n={env.action_space.n}, valid_masks={int(sum(masks))}",
        )
    except Exception as exc:
        return CheckResult(
            name="gym_env_contract",
            status="FAIL",
            detail=f"{type(exc).__name__}: {exc}",
        )


def check_output_paths() -> CheckResult:
    needed_dirs = [
        ROOT / "results",
        ROOT / "results" / "best_model",
        ROOT / "results" / "runs",
        ROOT / "results" / "eval_logs",
        ROOT / "logs",
    ]
    try:
        for d in needed_dirs:
            d.mkdir(parents=True, exist_ok=True)
            probe = d / ".write_probe.tmp"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
    except Exception as exc:
        return CheckResult(
            name="output_paths",
            status="FAIL",
            detail=f"{type(exc).__name__}: {exc}",
        )
    return CheckResult(
        name="output_paths",
        status="PASS",
        detail="results/ and logs/ are writable",
    )


def check_train_cli() -> CheckResult:
    cmd = [sys.executable, "-m", "rl.train_ppo", "--help"]
    rc, out, err = _run_cmd(cmd, ROOT)
    if rc != 0:
        return CheckResult(
            name="train_cli",
            status="FAIL",
            detail=err.strip() or "train_ppo --help failed",
        )
    needed_flags = [
        "--phase",
        "--timesteps",
        "--n_envs",
        "--task",
        "--phase1-eval-freq",
        "--phase1-n-eval-episodes",
        "--phase1-disable-eval-callback",
        "--phase1-grader-eval-freq-multiplier",
    ]
    missing = [f for f in needed_flags if f not in out]
    if missing:
        return CheckResult(
            name="train_cli",
            status="WARN",
            detail="Missing expected flags in help output: " + ", ".join(missing),
        )
    return CheckResult(
        name="train_cli",
        status="PASS",
        detail="train_ppo CLI flags detected",
    )


def check_config() -> CheckResult:
    try:
        import yaml

        cfg_path = ROOT / "rl" / "configs" / "ppo_easy.yaml"
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8-sig")) or {}
        hp = cfg.get("hyperparameters", {})
        tr = cfg.get("training", {})

        required_fields = [
            ("hyperparameters", "learning_rate"),
            ("hyperparameters", "n_steps"),
            ("hyperparameters", "batch_size"),
            ("training", "n_envs"),
            ("training", "seed"),
            ("training", "eval_freq"),
            ("training", "n_eval_episodes"),
        ]
        missing = []
        for section, key in required_fields:
            parent = hp if section == "hyperparameters" else tr
            if key not in parent:
                missing.append(f"{section}.{key}")
        if missing:
            return CheckResult(
                name="ppo_easy_config",
                status="FAIL",
                detail="Missing config fields: " + ", ".join(missing),
            )

        warnings = []
        if int(tr.get("eval_freq", 0)) < 2048:
            warnings.append("eval_freq is very low; may cause frequent pauses")
        if int(tr.get("n_eval_episodes", 0)) > 5:
            warnings.append("n_eval_episodes is high; callback cost may increase")

        if warnings:
            return CheckResult(
                name="ppo_easy_config",
                status="WARN",
                detail="; ".join(warnings),
            )
        return CheckResult(
            name="ppo_easy_config",
            status="PASS",
            detail="Phase 1 config fields are present and reasonable",
        )
    except Exception as exc:
        return CheckResult(
            name="ppo_easy_config",
            status="FAIL",
            detail=f"{type(exc).__name__}: {exc}",
        )


def check_torch_device() -> CheckResult:
    try:
        import torch

        if torch.cuda.is_available():
            return CheckResult(
                name="torch_device",
                status="PASS",
                detail=f"CUDA available ({torch.cuda.get_device_name(0)})",
            )
        return CheckResult(
            name="torch_device",
            status="WARN",
            detail="CUDA not available; CPU training is expected",
        )
    except Exception as exc:
        return CheckResult(
            name="torch_device",
            status="WARN",
            detail=f"torch device check skipped: {type(exc).__name__}: {exc}",
        )


def run_targeted_tests() -> CheckResult:
    test_cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_env.py",
        "tests/test_gym_wrapper.py",
        "tests/test_gym_wrapper_integration.py",
        "tests/test_feature_builder.py",
        "tests/test_action_mask.py",
        "tests/test_curriculum.py",
        "tests/test_rl_evaluate.py",
        "-q",
        "--tb=short",
    ]
    rc, out, err = _run_cmd(test_cmd, ROOT)
    if rc != 0:
        return CheckResult(
            name="targeted_tests",
            status="FAIL",
            detail=(out + "\n" + err).strip()[-1200:],
        )
    return CheckResult(
        name="targeted_tests",
        status="PASS",
        detail=out.strip().splitlines()[-1] if out.strip() else "targeted tests passed",
    )


def _print_results(results: list[CheckResult]) -> None:
    print("\n=== Pre-Train Checklist Results ===")
    for r in results:
        print(f"[{r.status}] {r.name}: {r.detail}")

    fail_count = sum(1 for r in results if r.status == "FAIL")
    warn_count = sum(1 for r in results if r.status == "WARN")
    print("\n=== Gate Decision ===")
    if fail_count > 0:
        print(f"NO-GO: {fail_count} failing check(s). Resolve failures before training.")
    else:
        print(
            f"GO: no failing checks. "
            f"{warn_count} warning(s) can be reviewed but do not block training."
        )


def _print_next_commands(args: argparse.Namespace) -> None:
    print("\n=== Recommended Phase 1 Commands (Manual) ===")
    print(
        "python -m rl.train_ppo "
        f"--phase 1 --task {PHASE1_TASK} "
        f"--timesteps {args.timesteps} --n_envs {args.n_envs} --seed {args.seed} "
        "--phase1-no-progress-bar "
        "--phase1-eval-freq 16384 "
        "--phase1-n-eval-episodes 2 "
        "--phase1-grader-eval-freq-multiplier 4"
    )
    print(
        "python rl/eval_grader.py "
        "--model results/best_model/phase1_final "
        f"--task {PHASE1_TASK} --episodes 20 --seed {args.seed}"
    )
    print(
        "python rl/plot_training.py "
        f"--task {PHASE1_TASK} --phase 1"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Pre-train checklist + GO/NO-GO gate")
    parser.add_argument("--run-tests", action="store_true", help="Run targeted RL tests")
    parser.add_argument("--timesteps", type=int, default=300000)
    parser.add_argument("--n-envs", "--n_envs", dest="n_envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json-out", default=None, help="Optional path to write JSON report")
    args = parser.parse_args()

    checks: list[Callable[[], CheckResult]] = [
        check_required_files,
        check_python_imports,
        check_compile,
        check_train_cli,
        check_config,
        check_env_contract,
        check_output_paths,
        check_torch_device,
    ]
    if args.run_tests:
        checks.append(run_targeted_tests)

    results = [fn() for fn in checks]
    _print_results(results)
    _print_next_commands(args)

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "go_no_go": "NO-GO" if any(r.status == "FAIL" for r in results) else "GO",
            "results": [asdict(r) for r in results],
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nJSON report written to: {out_path}")

    return 2 if any(r.status == "FAIL" for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
