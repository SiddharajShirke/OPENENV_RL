"""
Compatibility shim for legacy imports.

The high-level simulation orchestration now lives in app.engine.
This module re-exports the public runtime API so existing imports
from app.simulator continue to work unchanged.
"""

from __future__ import annotations

from app.engine import (
    DayResult,
    DaySimulator,
    LiveSimulationSession,
    SimulationAgentMode,
    SimulationRun,
    run_simulation,
)

__all__ = [
    "DayResult",
    "DaySimulator",
    "SimulationAgentMode",
    "SimulationRun",
    "LiveSimulationSession",
    "run_simulation",
]