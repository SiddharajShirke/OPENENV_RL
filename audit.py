import os
import sys
import json
import inspect
import requests
import numpy as np
import yaml
import gymnasium as gym

from stable_baselines3.common.env_checker import check_env
from sb3_contrib import MaskablePPO

def print_result(check_num, desc, status, detail=""):
    print(f"[CHECK {check_num}] {desc}\nSTATUS: {status}\nDETAIL: {detail}\n")

# B1
try:
    from app.models import (
        ServiceType, StageType, PriorityMode, ActionType,
        OfficerPool, QueueSnapshot, ObservationModel, ActionModel,
        RewardModel, EpisodeStateModel, StepInfoModel,
        SimulationConfig, TaskConfig, GraderResult,
        BenchmarkResult, LiveRunResult, EpisodeMetrics
    )
    print_result("B1", "All 17 Schemas Present", "PASS", "All 17 names resolve")
except Exception as e:
    print_result("B1", "All 17 Schemas Present", "FAIL", str(e))

# B2
try:
    fields = QueueSnapshot.model_fields
    assert 'total_pending' in fields, "total_pending missing"
    assert 'blocked_missing_docs' in fields, "blocked_missing_docs missing"
    assert 'active_cases' not in fields, "legacy field active_cases found"
    assert 'missing_docs_cases' not in fields, "legacy field found"

    m_fields = EpisodeMetrics.model_fields
    assert 'total_invalid_actions' in m_fields, "total_invalid_actions missing"
    print_result("B2", "Canonical Field Name Verification", "PASS", "Fields verified")
except Exception as e:
    print_result("B2", "Canonical Field Name Verification", "FAIL", str(e))

# B3
try:
    from app.simulator import SimulationAgentMode
    assert hasattr(SimulationAgentMode, 'BASELINE_POLICY'), "BASELINE_POLICY missing"
    assert hasattr(SimulationAgentMode, 'RANDOM'), "RANDOM missing"
    assert hasattr(SimulationAgentMode, 'LLM_AGENT'), "LLM_AGENT missing"
    assert hasattr(SimulationAgentMode, 'HEURISTIC'), "HEURISTIC missing"
    try:
        _ = SimulationAgentMode.baseline_policy
        print_result("B3", "Enum Casing Check", "FAIL", "lowercase alias exists")
    except AttributeError:
        print_result("B3", "Enum Casing Check", "PASS", "No lowercase alias")
except Exception as e:
    print_result("B3", "Enum Casing Check", "FAIL", str(e))

# C1
try:
    from app.env import GovWorkflowEnv
    env = GovWorkflowEnv(task_id="district_backlog_easy", seed=42)
    obs, info = env.reset(seed=42)
    assert isinstance(obs, dict), f"obs is {type(obs)}, expected dict"
    assert isinstance(info, dict), f"info is {type(info)}, expected dict"
    assert len(obs) > 0, "empty observation"
    print_result("C1", "reset() Returns (observation, info)", "PASS", "Valid dicts returned")
except Exception as e:
    print_result("C1", "reset() Returns (observation, info)", "FAIL", str(e))

# C2
try:
    from app.models import ActionModel, ActionType
    action = ActionModel(action_type=ActionType.ADVANCE_TIME)
    result = env.step(action)
    assert len(result) == 5, f"step() returned {len(result)} values, expected 5"
    obs2, reward, terminated, truncated, info2 = result
    assert isinstance(reward, float), f"reward type {type(reward)}"
    assert isinstance(terminated, bool), "terminated not bool"
    assert isinstance(truncated, bool), "truncated not bool"
    print_result("C2", "step() Returns (obs, reward, terminated, truncated, info)", "PASS", "Valid step signature")
except Exception as e:
    print_result("C2", "step() Returns (obs, reward, terminated, truncated, info)", "FAIL", str(e))

# C3 (Skipping dictionary check since MaskablePPO actually uses rl.gov_workflow_env for gym.Env spaces, doing that in J instead)
# Wait, let's just check the wrapper.
try:
    from rl.gov_workflow_env import GovWorkflowGymEnv
    genv = GovWorkflowGymEnv(task_id="district_backlog_easy", seed=42)
    gobs, _ = genv.reset(seed=42)
    def check_dtype(obs_dict, path="obs"):
        for k, v in obs_dict.items():
            if isinstance(v, np.ndarray):
                assert v.dtype == np.float32 or v.dtype == np.int64, f"FAIL: {path}.{k} dtype={v.dtype}"
            elif isinstance(v, dict):
                check_dtype(v, f"{path}.{k}")
    check_dtype(gobs)
    print_result("C3", "Observation Space Dtype (SB3 Requirement)", "PASS", "Wrapper dict is fine")
except Exception as e:
    print_result("C3", "Observation Space Dtype (SB3 Requirement)", "FAIL", str(e))

# C4
try:
    env1 = GovWorkflowEnv(task_id="district_backlog_easy", seed=42)
    env2 = GovWorkflowEnv(task_id="district_backlog_easy", seed=42)
    obs1, _ = env1.reset(seed=42)
    obs2, _ = env2.reset(seed=42)
    
    # Strip volatile message field before comparison (as in tests)
    obs1.last_action_explanation = ""
    obs2.last_action_explanation = ""
    obs1.episode_id = ""
    obs2.episode_id = ""

    assert json.dumps(obs1.model_dump(), sort_keys=True, default=str) == json.dumps(obs2.model_dump(), sort_keys=True, default=str), "Different observations"
    print_result("C4", "Determinism Check", "PASS", "Observations match")
except Exception as e:
    print_result("C4", "Determinism Check", "FAIL", str(e))

# C5
try:
    env_c5 = GovWorkflowEnv(task_id="district_backlog_easy", seed=42)
    obs, _ = env_c5.reset(seed=42)
    terminated = False
    truncated = False
    steps = 0
    max_steps = 500
    while not (terminated or truncated) and steps < max_steps:
        action = ActionModel(action_type=ActionType.ADVANCE_TIME)
        obs, reward, terminated, truncated, info = env_c5.step(action)
        steps += 1
    assert terminated or truncated, f"episode never ended after {max_steps} steps"
    print_result("C5", "Episode Termination Check", "PASS", f"ended at step {steps}")
except Exception as e:
    print_result("C5", "Episode Termination Check", "FAIL", str(e))

# D1
try:
    env_d1 = GovWorkflowEnv(task_id="district_backlog_easy", seed=42)
    obs, _ = env_d1.reset(seed=42)
    rewards = []
    for _ in range(20):
        action = ActionModel(action_type=ActionType.ADVANCE_TIME)
        obs, reward, term, trunc, info = env_d1.step(action)
        rewards.append(reward)
        if term or trunc: break
    nonzero = sum(1 for r in rewards if abs(r) > 1e-6)
    assert nonzero > len(rewards) * 0.5, f"Only {nonzero}/{len(rewards)} steps had nonzero reward"
    print_result("D1", "Reward is Dense", "PASS", f"{nonzero}/{len(rewards)} steps nonzero")
except Exception as e:
    print_result("D1", "Reward is Dense", "FAIL", str(e))

# D2
try:
    for r in rewards:
        assert -100 <= r <= 100, f"reward {r} outside [-100, 100]"
    print_result("D2", "Reward Range Sanity Check", "PASS", "Rewards in bounds")
except Exception as e:
    print_result("D2", "Reward Range Sanity Check", "FAIL", str(e))

# D3
try:
    from app.models import ServiceType
    env_d3 = GovWorkflowEnv(task_id="district_backlog_easy", seed=42)
    obs, _ = env_d3.reset(seed=42)
    # Using a valid enum but perhaps invalid context to cause penalty
    # The framework doesn't allow 'nonexistent' string if it's an Enum, so let's use valid enum but no cases.
    bad_action = ActionModel(action_type=ActionType.ESCALATE_SERVICE, service_target=ServiceType.PASSPORT)
    obs, reward, term, trunc, info = env_d3.step(bad_action)
    assert reward <= 0, f"invalid action produced positive reward {reward}"
    print_result("D3", "Invalid Action Penalty Fires", "PASS", f"reward={reward:.3f}")
except Exception as e:
    print_result("D3", "Invalid Action Penalty Fires", "FAIL", str(e))

# E1
try:
    from app.tasks import get_task
    for task_id in ["district_backlog_easy", "mixed_urgency_medium", "cross_department_hard"]:
        cfg = get_task(task_id)
        assert cfg.seed is not None, f"{task_id} has no seed"
        assert cfg.max_days > 0, f"{task_id} max_days={cfg.max_days}"
    print_result("E1", "All 3 Tasks Loadable", "PASS", "All config loaded")
except Exception as e:
    print_result("E1", "All 3 Tasks Loadable", "FAIL", str(e))

# E2
try:
    from app.graders import grade_episode
    for task_id in ["district_backlog_easy", "mixed_urgency_medium", "cross_department_hard"]:
        env_e2 = GovWorkflowEnv(task_id=task_id, seed=42)
        obs, _ = env_e2.reset(seed=42)
        terminated = truncated = False
        while not (terminated or truncated):
            obs, reward, terminated, truncated, info = env_e2.step(ActionModel(action_type=ActionType.ADVANCE_TIME))
        episode_state = env_e2.state()
        score_res = grade_episode(episode_state)
        assert isinstance(score_res.score, float), f"grader returned {type(score_res.score)}"
        assert 0.0 <= score_res.score <= 1.0, f"score={score_res.score} outside [0.0, 1.0]"
    print_result("E2", "Graders Return [0.0, 1.0]", "PASS", "Valid scores returned")
except Exception as e:
    print_result("E2", "Graders Return [0.0, 1.0]", "FAIL", str(e))

# E3
try:
    scores = []
    for _ in range(2):
        env_e3 = GovWorkflowEnv(task_id="district_backlog_easy", seed=42)
        obs, _ = env_e3.reset(seed=42)
        terminated = truncated = False
        while not (terminated or truncated):
            obs, r, terminated, truncated, info = env_e3.step(ActionModel(action_type=ActionType.ADVANCE_TIME))
        scores.append(grade_episode(env_e3.state()).score)
    assert scores[0] == scores[1], f"grader is non-deterministic: {scores}"
    print_result("E3", "Grader Scores Are Deterministic", "PASS", f"score={scores[0]:.4f} both runs")
except Exception as e:
    print_result("E3", "Grader Scores Are Deterministic", "FAIL", str(e))

# F1
try:
    from app.state_machine import StateMachine, StageType, WorkflowAction
    sm = StateMachine()
    stages = [StageType.SUBMISSION, StageType.DOCUMENT_VERIFICATION, StageType.FIELD_VERIFICATION, StageType.APPROVAL, StageType.ISSUANCE]
    for i in range(len(stages) - 1):
        current = stages[i]
        next_stage = stages[i + 1]
        result = sm.transition(current, WorkflowAction.ADVANCE)
        assert result == next_stage, f"{current} -> {result}, expected {next_stage}"
    print_result("F1", "All Legal Transitions Work", "PASS", "Transitions validated")
except Exception as e:
    print_result("F1", "All Legal Transitions Work", "FAIL", str(e))

# F2
try:
    assert sm.is_terminal(StageType.ISSUANCE) == True, "issuance not recognized as terminal"
    assert sm.is_terminal(StageType.SUBMISSION) == False, "submission wrongly marked terminal"
    print_result("F2", "Terminal State Recognized", "PASS", "Terminal states correct")
except Exception as e:
    print_result("F2", "Terminal State Recognized", "FAIL", str(e))

# G1
try:
    import app.simulator as sim_module
    source = inspect.getfile(sim_module.LiveSimulationSession)
    assert 'engine' in source.lower(), f"LiveSimulationSession defined in {source}, not engine.py"
    print_result("G1", "simulator.py Is a Pure Shim", "PASS", "Shim logic confirmed")
except Exception as e:
    print_result("G1", "simulator.py Is a Pure Shim", "FAIL", str(e))

# G2
try:
    from app.simulator import LiveSimulationSession, SimulationAgentMode, run_simulation
    assert callable(run_simulation), "run_simulation not callable"
    assert callable(LiveSimulationSession), "LiveSimulationSession not callable"
    print_result("G2", "All 3 Engine Exports Importable", "PASS", "Exports valid")
except Exception as e:
    print_result("G2", "All 3 Engine Exports Importable", "FAIL", str(e))

# G3
try:
    session = LiveSimulationSession(
        task_id="district_backlog_easy",
        agent_mode=SimulationAgentMode.BASELINE_POLICY,
        seed=42,
        max_steps=10
    )
    start_info = session.start_line()
    assert isinstance(start_info, str), "start_line() did not return str"
    step_result, _, _ = session.step_once()
    assert "observation" in step_result, "step_once missing 'observation'"
    assert "reward" in step_result, "step_once missing 'reward'"
    print_result("G3", "LiveSimulationSession Full Lifecycle", "PASS", "Lifecycle valid")
    session.close()
except Exception as e:
    print_result("G3", "LiveSimulationSession Full Lifecycle", "FAIL", str(e))

# H2 / H3
# We will do H checks via curl/pytest in bash to test the live server.

# I1
try:
    from app.baselines import (
        random_policy,
        backlog_clearance_policy as baseline_policy,
        greedy_sla_policy,
        fairness_aware_policy,
    )
    for name, fn in [
        ("random_policy", random_policy),
        ("baseline_policy", baseline_policy),
        ("greedy_sla_policy", greedy_sla_policy),
        ("fairness_aware_policy", fairness_aware_policy),
    ]:
        assert callable(fn), f"{name} is not callable"
    print_result("I1", "All 4 Policies Are Callable", "PASS", "Policies callable")
except Exception as e:
    print_result("I1", "All 4 Policies Are Callable", "FAIL", str(e))

# I2
try:
    from app.baselines import greedy_sla_policy
    env_i2 = GovWorkflowEnv(task_id="district_backlog_easy", seed=42)
    obs_i2, _ = env_i2.reset(seed=42)
    action_i2 = greedy_sla_policy(obs_i2)
    assert isinstance(action_i2, ActionModel), f"policy returned {type(action_i2)}"
    print_result("I2", "Policy Returns Valid Action", "PASS", f"action_type={action_i2.action_type}")
except Exception as e:
    print_result("I2", "Policy Returns Valid Action", "FAIL", str(e))

# J1
try:
    env_j1 = GovWorkflowGymEnv(task_id="district_backlog_easy", seed=42)
    assert hasattr(env_j1, 'observation_space'), "no observation_space"
    assert hasattr(env_j1, 'action_space'), "no action_space"
    print_result("J1", "Gymnasium API Compliance", "PASS", "Spaces defined")
except Exception as e:
    print_result("J1", "Gymnasium API Compliance", "FAIL", str(e))

# J2
try:
    obs, _ = env_j1.reset(seed=42)
    assert hasattr(env_j1, 'action_masks'), "action_masks() method missing"
    masks = env_j1.action_masks()
    assert hasattr(masks, '__len__'), "action_masks() must return array-like"
    assert len(masks) == env_j1.action_space.n, f"mask length {len(masks)} != action_space.n {env_j1.action_space.n}"
    print_result("J2", "action_masks() Method Required by MaskablePPO", "PASS", f"n={len(masks)}")
except Exception as e:
    print_result("J2", "action_masks() Method Required by MaskablePPO", "FAIL", str(e))

# J3
try:
    check_env(env_j1, warn=True)
    print_result("J3", "SB3 VecEnv Compatibility", "PASS", "check_env passed")
except Exception as e:
    print_result("J3", "SB3 VecEnv Compatibility", "FAIL", str(e))

# J4
try:
    model = MaskablePPO("MlpPolicy", env_j1, verbose=0, seed=42)
    print_result("J4", "MaskablePPO Can Initialize", "PASS", "Model initialized")
except Exception as e:
    print_result("J4", "MaskablePPO Can Initialize", "FAIL", str(e))

# J5
try:
    obs, _ = env_j1.reset(seed=42)
    for step in range(10):
        masks = env_j1.action_masks()
        valid_actions = [i for i, m in enumerate(masks) if m]
        action = valid_actions[0] if valid_actions else 0
        obs, reward, terminated, truncated, info = env_j1.step(action)
        if terminated or truncated:
            obs, _ = env_j1.reset(seed=42)
    print_result("J5", "10-Step Rollout Without Crash", "PASS", "Rollout passed")
except Exception as e:
    print_result("J5", "10-Step Rollout Without Crash", "FAIL", str(e))

# M1
try:
    with open("openenv.yaml", "r") as f:
        config = yaml.safe_load(f)
    assert "tasks" in config, "openenv.yaml missing 'tasks' key"
    task_ids = [t["id"] for t in config["tasks"]]
    for required in ["district_backlog_easy", "mixed_urgency_medium", "cross_department_hard"]:
        assert required in task_ids, f"{required} missing from openenv.yaml"
    print_result("M1", "YAML Loads and Contains All 3 Tasks", "PASS", f"{len(task_ids)} tasks registered")
except Exception as e:
    print_result("M1", "YAML Loads and Contains All 3 Tasks", "FAIL", str(e))

