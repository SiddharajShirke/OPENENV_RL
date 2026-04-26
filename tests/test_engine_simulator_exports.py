import importlib
import sys

from app.engine import (
    DayResult as EngineDayResult,
    DaySimulator as EngineDaySimulator,
    LiveSimulationSession as EngineLiveSimulationSession,
    SimulationAgentMode as EngineSimulationAgentMode,
    run_simulation as engine_run_simulation,
)
from app.simulator import (
    DayResult as ShimDayResult,
    DaySimulator as ShimDaySimulator,
    LiveSimulationSession as ShimLiveSimulationSession,
    SimulationAgentMode as ShimSimulationAgentMode,
    run_simulation as shim_run_simulation,
)


def test_simulator_shim_reexports_engine_symbols():
    assert ShimDayResult is EngineDayResult
    assert ShimDaySimulator is EngineDaySimulator
    assert ShimLiveSimulationSession is EngineLiveSimulationSession
    assert ShimSimulationAgentMode is EngineSimulationAgentMode
    assert shim_run_simulation is engine_run_simulation


def test_day_result_has_runtime_fields():
    result = EngineDayResult()
    assert hasattr(result, "digital_arrivals")
    assert hasattr(result, "newly_blocked_missing")
    assert hasattr(result, "newly_unblocked_enrich")


def test_import_env_then_simulator_succeeds():
    for name in ["app.engine", "app.simulator", "app.env"]:
        sys.modules.pop(name, None)

    env_mod = importlib.import_module("app.env")
    sim_mod = importlib.import_module("app.simulator")

    assert hasattr(env_mod, "GovWorkflowEnv")
    assert hasattr(sim_mod, "LiveSimulationSession")
    assert hasattr(sim_mod, "run_simulation")