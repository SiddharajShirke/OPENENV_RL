# scripts/

Utility scripts for run, validation, and benchmarking.

- `run_local.py`: launch local API server
- `validate_env.py`: local environment validation checks
- `validate-submission.sh`: deployment validation flow
- `pre_deploy_e2e.ps1`: Windows pre-deploy gate for end-to-end readiness before Docker build and release
- `benchmark_ladder.py`: compare heuristic and RL agents
- `smoke_test.py`: quick endpoint sanity checks

## Pre-deploy E2E gate (Windows)

Run a full readiness pass before release deployment:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\pre_deploy_e2e.ps1
```

Useful options:

```powershell
# Faster pass (skips extended regression test bundle)
powershell -ExecutionPolicy Bypass -File .\scripts\pre_deploy_e2e.ps1 -Quick

# Skip Docker checks when only validating local test readiness
powershell -ExecutionPolicy Bypass -File .\scripts\pre_deploy_e2e.ps1 -SkipDockerBuild -SkipDockerRuntime

# Use a specific Python interpreter
powershell -ExecutionPolicy Bypass -File .\scripts\pre_deploy_e2e.ps1 -PythonPath .\.venv313\Scripts\python.exe
```
