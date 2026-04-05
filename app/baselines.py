from __future__ import annotations
from collections.abc import Callable
from app.env import GovWorkflowEnv
from app.graders import grade_episode
from app.models import ActionModel, ActionType, ObservationModel, PriorityMode, ServiceType

PolicyFn = Callable[[ObservationModel], ActionModel]


def _service_with_max(attr: str, obs: ObservationModel) -> ServiceType | None:
    ranked = sorted(obs.queue_snapshots,
                    key=lambda s: (getattr(s, attr), s.active_cases, s.service.value), reverse=True)
    return ranked[0].service if ranked and getattr(ranked[0], attr) > 0 else None


def urgent_first_policy(obs: ObservationModel) -> ActionModel:
    if obs.priority_mode != PriorityMode.URGENT_FIRST:
        return ActionModel(action_type=ActionType.SET_PRIORITY_MODE, priority_mode=PriorityMode.URGENT_FIRST)
    target = _service_with_max("missing_docs_cases", obs)
    if target:
        return ActionModel(action_type=ActionType.REQUEST_MISSING_DOCUMENTS, service=target)
    return ActionModel(action_type=ActionType.ADVANCE_TIME)


def oldest_first_policy(obs: ObservationModel) -> ActionModel:
    if obs.priority_mode != PriorityMode.OLDEST_FIRST:
        return ActionModel(action_type=ActionType.SET_PRIORITY_MODE, priority_mode=PriorityMode.OLDEST_FIRST)
    return ActionModel(action_type=ActionType.ADVANCE_TIME)


def backlog_clearance_policy(obs: ObservationModel) -> ActionModel:
    if obs.priority_mode != PriorityMode.BACKLOG_CLEARANCE:
        return ActionModel(action_type=ActionType.SET_PRIORITY_MODE, priority_mode=PriorityMode.BACKLOG_CLEARANCE)
    if obs.officer_pool.reserve_officers > 0:
        target = _service_with_max("active_cases", obs)
        if target:
            return ActionModel(action_type=ActionType.ASSIGN_CAPACITY, service=target, officer_delta=1)
    target = _service_with_max("missing_docs_cases", obs)
    if target:
        return ActionModel(action_type=ActionType.REQUEST_MISSING_DOCUMENTS, service=target)
    hot  = sorted(obs.queue_snapshots, key=lambda s: s.active_cases, reverse=True)
    cold = sorted(obs.queue_snapshots, key=lambda s: s.active_cases)
    if hot and cold and hot[0].active_cases - cold[0].active_cases >= 3:
        src, tgt = cold[0].service, hot[0].service
        if src != tgt and obs.officer_pool.allocations.get(src, 0) > 1:
            return ActionModel(action_type=ActionType.REALLOCATE_OFFICERS,
                               service=src, target_service=tgt, officer_delta=1)
    return ActionModel(action_type=ActionType.ADVANCE_TIME)


POLICIES: dict[str, PolicyFn] = {
    "urgent_first":      urgent_first_policy,
    "oldest_first":      oldest_first_policy,
    "backlog_clearance": backlog_clearance_policy,
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