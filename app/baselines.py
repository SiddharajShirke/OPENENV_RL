from __future__ import annotations
from collections.abc import Callable
from types import SimpleNamespace
from app.env import GovWorkflowEnv
from app.graders import grade_episode
from app.models import ActionModel, ActionType, ObservationModel, PriorityMode, ServiceType

PolicyFn = Callable[[ObservationModel], ActionModel]


def _snapshots(obs: ObservationModel):
    """Return queue snapshots as a list regardless of Phase 1 (list) or Phase 2 (dict)."""
    qs = obs.queue_snapshots
    if isinstance(qs, dict):
        return list(qs.values())
    return list(qs)


def _service_attr(q, *attrs):
    """Return the first attribute that exists on a QueueSnapshot (Phase 1 vs Phase 2 names)."""
    for attr in attrs:
        val = getattr(q, attr, None)
        if val is not None:
            return val
    return 0


def _service_name(q) -> ServiceType:
    """Return ServiceType regardless of Phase 1 (.service) or Phase 2 (.service_type)."""
    return getattr(q, "service_type", None) or getattr(q, "service", None)


def _service_with_max(obs: ObservationModel, *attrs) -> ServiceType | None:
    snaps = _snapshots(obs)
    ranked = sorted(snaps, key=lambda s: _service_attr(s, *attrs), reverse=True)
    if ranked and _service_attr(ranked[0], *attrs) > 0:
        return _service_name(ranked[0])
    return None


def _reserve_officers(obs: ObservationModel) -> int:
    pool = obs.officer_pool
    # Phase 2: idle_officers property
    if hasattr(pool, "idle_officers"):
        return int(pool.idle_officers)
    # Phase 1 fallback
    return int(getattr(pool, "reserve_officers", 0))


def _alloc_for(obs: ObservationModel, service: ServiceType) -> int:
    pool = obs.officer_pool
    # Phase 2 uses 'allocated'; Phase 1 used 'allocations'
    alloc_dict = getattr(pool, "allocated", None) or getattr(pool, "allocations", {})
    raw = alloc_dict.get(service)
    if raw is None:
        raw = alloc_dict.get(service.value if hasattr(service, "value") else str(service), 0)
    return int(raw or 0)


def urgent_first_policy(obs: ObservationModel) -> ActionModel:
    target = _service_with_max(obs, "urgent_pending", "urgent_cases")
    if target:
        return ActionModel(action_type=ActionType.REQUEST_MISSING_DOCUMENTS, service_target=target)
    return ActionModel(action_type=ActionType.ADVANCE_TIME)


def oldest_first_policy(obs: ObservationModel) -> ActionModel:
    return ActionModel(action_type=ActionType.ADVANCE_TIME)


def backlog_clearance_policy(obs: ObservationModel) -> ActionModel:
    snaps = _snapshots(obs)

    # Assign idle officers to the most backlogged service
    if _reserve_officers(obs) > 0:
        target = _service_with_max(obs, "total_pending", "active_cases")
        if target:
            return ActionModel(
                action_type=ActionType.ASSIGN_CAPACITY,
                service_target=target,
                capacity_assignment={target.value: 1},
            )

    # Clear missing-doc bottlenecks
    target = _service_with_max(obs, "blocked_missing_docs", "missing_docs_cases")
    if target:
        return ActionModel(action_type=ActionType.REQUEST_MISSING_DOCUMENTS, service_target=target)

    # Reallocate from least-loaded to most-loaded
    if len(snaps) >= 2:
        hot = sorted(snaps, key=lambda s: _service_attr(s, "total_pending", "active_cases"), reverse=True)
        cold = sorted(snaps, key=lambda s: _service_attr(s, "total_pending", "active_cases"))
        hot_svc = _service_name(hot[0])
        cold_svc = _service_name(cold[0])
        hot_load = _service_attr(hot[0], "total_pending", "active_cases")
        cold_load = _service_attr(cold[0], "total_pending", "active_cases")
        if (
            hot_svc and cold_svc and hot_svc != cold_svc
            and hot_load - cold_load >= 3
            and _alloc_for(obs, cold_svc) > 1
        ):
            return ActionModel(
                action_type=ActionType.REALLOCATE_OFFICERS,
                service_target=cold_svc,
                reallocation_delta={cold_svc.value: -1, hot_svc.value: 1},
            )

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
    # Return a SimpleNamespace so attribute access (result.score) works in main.py
    return SimpleNamespace(
        task_id=task_id,
        policy=policy_name,
        seed=state.seed,
        reward_sum=round(reward_sum, 4),
        score=float(grade.score),
        grader=grade.grader_name,
        metrics=grade.metrics,
        steps=int(state.total_steps),
        completed=int(state.total_completed),
        backlog=int(state.total_backlog),
    )