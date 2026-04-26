def test_legacy_simulator_import_path_is_live():
    from app.simulator import LiveSimulationSession, SimulationAgentMode, run_simulation

    assert LiveSimulationSession is not None
    assert SimulationAgentMode is not None
    assert callable(run_simulation)
    