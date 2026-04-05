from __future__ import annotations
from app.models import RewardModel


def compute_reward(
    *,
    stage_advances: int,
    completions: int,
    active_backlog: int,
    new_sla_breaches: int,
    fairness_gap: float,
    fairness_threshold: float | None,
    invalid_action: bool,
    idle_capacity: int,
) -> RewardModel:
    progress_reward     = 0.7  * stage_advances
    completion_reward   = 4.0  * completions
    waiting_penalty     = 0.04 * active_backlog
    sla_penalty         = 1.5  * new_sla_breaches
    unfairness_excess   = max(0.0, fairness_gap - (fairness_threshold or 0.0)) \
                          if fairness_threshold is not None \
                          else max(0.0, fairness_gap - 0.4)
    fairness_penalty        = 2.0 * unfairness_excess
    invalid_action_penalty  = 1.5 if invalid_action else 0.0
    idle_capacity_penalty   = 0.05 * idle_capacity

    total_reward = (
        progress_reward + completion_reward
        - waiting_penalty - sla_penalty
        - fairness_penalty - invalid_action_penalty
        - idle_capacity_penalty
    )
    return RewardModel(
        total_reward=round(total_reward, 4),
        progress_reward=round(progress_reward, 4),
        completion_reward=round(completion_reward, 4),
        waiting_penalty=round(waiting_penalty, 4),
        sla_penalty=round(sla_penalty, 4),
        fairness_penalty=round(fairness_penalty, 4),
        invalid_action_penalty=round(invalid_action_penalty, 4),
        idle_capacity_penalty=round(idle_capacity_penalty, 4),
    )