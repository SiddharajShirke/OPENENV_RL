# Gov Workflow OpenEnv — Phase 1 & 2 Validation Checklist

## How to Run Tests

```bash
# 1. Install test dependencies (run once)
pip install pytest pytest-cov fastapi httpx

# 2. Run all tests at once
bash run_all_tests.sh

# 3. Run Phase 1 only
bash run_all_tests.sh --unit-only

# 4. Run API tests only (no server needed — uses TestClient)
bash run_all_tests.sh --api-only

# 5. Quick smoke test (skip coverage)
bash run_all_tests.sh --fast

# 6. Individual module
pytest tests/test_phase1_models.py -v
pytest tests/test_phase1_sector_and_tasks.py -v
pytest tests/test_phase1_event_engine.py -v
pytest tests/test_phase1_signal_computer.py -v
pytest tests/test_phase2_env_integration.py -v
pytest tests/test_phase2_simulator.py -v
pytest tests/test_phase2_api.py -v
pytest tests/test_phase2_graders.py -v
```

---

## Test Files Summary

| File | Covers | Tests |
|---|---|---|
| `test_phase1_models.py` | models.py enums, OfficerPool, ApplicationCase, ObservationModel, ActionModel, RewardModel | 40+ |
| `test_phase1_sector_and_tasks.py` | sector_profiles.py, tasks.py | 45+ |
| `test_phase1_event_engine.py` | event_engine.py determinism, effects, describe | 35+ |
| `test_phase1_signal_computer.py` | signal_computer.py all 7 signals | 30+ |
| `test_phase2_env_integration.py` | env.py reset/step/state API, action dispatch, determinism | 40+ |
| `test_phase2_simulator.py` | simulator.py DaySimulator, DayResult, snapshots, case gen | 25+ |
| `test_phase2_api.py` | FastAPI /health /reset /step /state /grade /sessions | 40+ |
| `test_phase2_graders.py` | graders.py scoring all 3 tasks, bounds, determinism | 20+ |
| `conftest.py` | Shared fixtures for all modules | — |

---

## Phase Completion Status

### Phase 1 — Foundation ✅
- [x] `models.py` — All 15 Pydantic models typed
- [x] `sector_profiles.py` — 7 sector profiles with realistic values
- [x] `tasks.py` — 3 benchmark tasks with deterministic seeds
- [x] `event_engine.py` — Deterministic daily event system
- [x] `signal_computer.py` — 7 normalized compressed state signals

### Phase 2 — Environment Core ✅ (from your uploaded files)
- [x] `env.py` — GovWorkflowEnv with reset/step/state/apply_action
- [x] `simulator.py` — DaySimulator, DayResult, case lifecycle
- [x] `state_machine.py` — advance_case, unblock_missing_docs, field verification
- [x] `reward.py` — Dense shaped reward with 10 components
- [x] `graders.py` — Deterministic scores per task
- [x] `main.py` — FastAPI with /health /reset /step /state /grade /sessions

---

## Expected Test Output (passing)

```
tests/test_phase1_models.py::TestEnums::test_service_types_count PASSED
tests/test_phase1_models.py::TestEnums::test_all_service_types_present PASSED
tests/test_phase1_models.py::TestOfficerPool::test_idle_officers_calculation PASSED
...
tests/test_phase2_api.py::TestHealth::test_health_returns_200 PASSED
tests/test_phase2_api.py::TestReset::test_reset_returns_session_id PASSED
...
======== 275 passed in 12.4s ========
```

---

## Common Failures and Fixes

| Error | Cause | Fix |
|---|---|---|
| `ImportError: cannot import name 'DocEnrichmentType'` | Phase 2 env.py uses Phase 2 models not yet in models.py | Add DocEnrichmentType enum to models.py |
| `KeyError: 'income_certificate'` in tasks | ServiceType enum key vs string mismatch | Use ServiceType enum as dict key not string |
| `AttributeError: OfficerPool has no idle_officers` | Property missing | Ensure @property idle_officers defined in OfficerPool |
| `AttributeError: ApplicationCase has no sla_risk` | Property missing | Ensure @property sla_risk defined |
| `pytest: no tests ran` | Tests folder missing __init__.py | Create empty `tests/__init__.py` |
| `ModuleNotFoundError: app.sector_profiles` | File not created yet | Create sector_profiles.py from Phase 1 delivery |
| `409 Conflict on step after 200 steps` | Episode already terminated | Normal — episode correctly enforces termination |
| Grade score outside [0,1] | Grader division by zero | Guard with `max(1, total_cases)` in grader |
