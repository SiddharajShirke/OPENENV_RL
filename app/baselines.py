from __future__ import annotations
from collections.abc import Callable
from app.env import GovWorkflowEnv
from app.graders import grade_episode
from app.models import ActionModel, ActionType, ObservationModel, PriorityMode, ServiceType

PolicyFn = Callable[[ObservationModel], ActionModel]


def _service_with_max(attr: str, obs: ObservationModel) -> ServiceType | None:
    queue_snaps = obs.queue_snapshots.values() if isinstance(obs.queue_snapshots, dict) else obs.queue_snapshots
    v2_attr_map = {
        "active_cases": "total_pending",
        "missing_docs_cases": "blocked_missing_docs"
    }
    def get_val(s):
        return getattr(s, attr, getattr(s, v2_attr_map.get(attr, attr), 0))
    ranked = sorted(queue_snaps,
                    key=lambda s: (get_val(s), getattr(s, "active_cases", getattr(s, "total_pending", 0)), getattr(s, "service_type", getattr(s, "service", "")).value), reverse=True)
    return (ranked[0].service_type if hasattr(ranked[0], "service_type") else ranked[0].service) if ranked and get_val(ranked[0]) > 0 else None


def greedy_sla_policy(obs: ObservationModel) -> ActionModel:
    target = _service_with_max("blocked_missing_docs", obs)
    if target:
        return ActionModel(action_type=ActionType.REQUEST_MISSING_DOCUMENTS, service_target=target)
    return ActionModel(action_type=ActionType.ADVANCE_TIME)


def oldest_first_policy(obs: ObservationModel) -> ActionModel:
    return ActionModel(action_type=ActionType.ADVANCE_TIME)


def backlog_clearance_policy(obs: ObservationModel) -> ActionModel:
    idle_officers = getattr(obs.officer_pool, "idle_officers", getattr(obs.officer_pool, "reserve_officers", 0))
    if idle_officers > 0:
        target = _service_with_max("total_pending", obs)
        if target:
            return ActionModel(action_type=ActionType.ASSIGN_CAPACITY, service_target=target, capacity_assignment={target.value: 1})
    target = _service_with_max("blocked_missing_docs", obs)
    if target:
        return ActionModel(action_type=ActionType.REQUEST_MISSING_DOCUMENTS, service_target=target)
    
    queue_snaps = obs.queue_snapshots.values() if isinstance(obs.queue_snapshots, dict) else obs.queue_snapshots
    hot  = sorted(queue_snaps, key=lambda s: getattr(s, "active_cases", getattr(s, "total_pending", 0)), reverse=True)
    cold = sorted(queue_snaps, key=lambda s: getattr(s, "active_cases", getattr(s, "total_pending", 0)))
    if hot and cold and getattr(hot[0], "active_cases", getattr(hot[0], "total_pending", 0)) - getattr(cold[0], "active_cases", getattr(cold[0], "total_pending", 0)) >= 3:
        src = hot[0].service_type if hasattr(hot[0], "service_type") else hot[0].service
        tgt = cold[0].service_type if hasattr(cold[0], "service_type") else cold[0].service
        allocs = getattr(obs.officer_pool, "allocated", getattr(obs.officer_pool, "allocations", {}))
        if src != tgt and allocs.get(src, allocs.get(src.value if hasattr(src, "value") else src, 0)) > 1:
            return ActionModel(action_type=ActionType.REALLOCATE_OFFICERS,
                               reallocation_delta={src.value if hasattr(src, "value") else src: -1, tgt.value if hasattr(tgt, "value") else tgt: 1})
    return ActionModel(action_type=ActionType.ADVANCE_TIME)

def random_policy(obs: ObservationModel) -> ActionModel:
    import random
    return ActionModel(action_type=ActionType.ADVANCE_TIME)

urgent_first_policy = greedy_sla_policy
fairness_aware_policy = backlog_clearance_policy

POLICIES: dict[str, PolicyFn] = {
    "urgent_first":      greedy_sla_policy,
    "oldest_first":      oldest_first_policy,
    "backlog_clearance": backlog_clearance_policy,
    "random_policy":     random_policy,
    "greedy_sla_policy": greedy_sla_policy,
    "fairness_aware_policy": fairness_aware_policy,
}


def run_policy_episode(task_id: str, policy_name: str, seed: int | None = None, max_steps: int = 500) -> dict:
    env = GovWorkflowEnv(task_id=task_id)
    obs, _ = env.reset(seed=seed)
    policy = POLICIES[policy_name]
    reward_sum = 0.0
    for _ in range(max_steps):
        action = policy(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        reward_sum += reward
        if terminated or truncated:
            break
    state = env.state()
    grade = grade_episode(state)
    return {
        "task_id":    task_id,
        "policy":     policy_name,
        "seed":       state.seed,
        "reward_sum": round(reward_sum, 4),
        "score":      grade.score,
        "grader":     grade.grader_name,
        "metrics":    grade.metrics,
        "steps":      state.total_steps,
        "completed":  state.total_completed,
        "backlog":    state.total_backlog,
    }
