#!/usr/bin/env bash
# ───────────────────────────────────────────────────────────────────
# run_all_tests.sh  — Full Phase 1 + Phase 2 validation suite
# Usage: bash run_all_tests.sh [--fast] [--api-only] [--unit-only]
# ───────────────────────────────────────────────────────────────────
set -e

# Colors
GREEN="\033[0;32m"
YELLOW="\033[1;33m"
RED="\033[0;31m"
NC="\033[0m"

FAST=false
API_ONLY=false
UNIT_ONLY=false

for arg in "$@"; do
  case $arg in
    --fast)       FAST=true ;;
    --api-only)   API_ONLY=true ;;
    --unit-only)  UNIT_ONLY=true ;;
  esac
done

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Gov Workflow OpenEnv — Test Suite     ${NC}"
echo -e "${GREEN}========================================${NC}"

# ── Phase 1: Unit Tests ───────────────────────────────────────────
if [ "$API_ONLY" = false ]; then
  echo -e "\n${YELLOW}[Phase 1] Running model schema tests...${NC}"
  pytest tests/test_phase1_models.py -v --tb=short

  echo -e "\n${YELLOW}[Phase 1] Running sector profile + task tests...${NC}"
  pytest tests/test_phase1_sector_and_tasks.py -v --tb=short

  echo -e "\n${YELLOW}[Phase 1] Running event engine tests...${NC}"
  pytest tests/test_phase1_event_engine.py -v --tb=short

  echo -e "\n${YELLOW}[Phase 1] Running signal computer tests...${NC}"
  pytest tests/test_phase1_signal_computer.py -v --tb=short
fi

# ── Phase 2: Integration Tests ────────────────────────────────────
if [ "$UNIT_ONLY" = false ]; then
  echo -e "\n${YELLOW}[Phase 2] Running env integration tests...${NC}"
  pytest tests/test_phase2_env_integration.py -v --tb=short

  echo -e "\n${YELLOW}[Phase 2] Running simulator tests...${NC}"
  pytest tests/test_phase2_simulator.py -v --tb=short

  echo -e "\n${YELLOW}[Phase 2] Running API endpoint tests (TestClient)...${NC}"
  pytest tests/test_phase2_api.py -v --tb=short
fi

# ── Summary ───────────────────────────────────────────────────────
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}  All test suites completed.${NC}"
echo -e "${GREEN}========================================${NC}"

# Full coverage report
if [ "$FAST" = false ]; then
  echo -e "\n${YELLOW}Running full coverage report...${NC}"
  pytest tests/ \
    --cov=app \
    --cov-report=term-missing \
    --cov-report=html:htmlcov \
    -q 2>/dev/null || true
  echo -e "${GREEN}Coverage report written to htmlcov/index.html${NC}"
fi
