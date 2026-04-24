# Gov Workflow OpenEnv -- Test Suite

## Setup
```bash
pip install pytest pytest-cov fastapi httpx
```

## Run All Tests (275+)
```bash
bash run_all_tests.sh
```

## Run Individual Suites
```bash
pytest tests/test_phase1_models.py -v
pytest tests/test_phase1_sector_and_tasks.py -v
pytest tests/test_phase1_event_engine.py -v
pytest tests/test_phase1_signal_computer.py -v
pytest tests/test_phase2_env_integration.py -v
pytest tests/test_phase2_simulator.py -v
pytest tests/test_phase2_api.py -v
pytest tests/test_phase2_graders.py -v
```

## Flags
```bash
bash run_all_tests.sh --unit-only   # Phase 1 only (models, sectors, events, signals)
bash run_all_tests.sh --api-only    # API + integration + grader tests only
bash run_all_tests.sh --fast        # Skip coverage report
```

## File Map
| File                              | Covers                                      | Tests |
|-----------------------------------|---------------------------------------------|-------|
| test_phase1_models.py             | Pydantic schemas, enums, SLA math           | 40+   |
| test_phase1_sector_and_tasks.py   | Sector profiles, task configs, seeds        | 45+   |
| test_phase1_event_engine.py       | Determinism, scenario scaling, event effects| 35+   |
| test_phase1_signal_computer.py    | 7 compressed state signals                  | 30+   |
| test_phase2_env_integration.py    | reset/step/state API, action dispatch       | 40+   |
| test_phase2_simulator.py          | DaySimulator, case lifecycle, snapshots     | 25+   |
| test_phase2_api.py                | FastAPI endpoints via TestClient            | 40+   |
| test_phase2_graders.py            | Deterministic grading, score bounds [0,1]   | 20+   |
