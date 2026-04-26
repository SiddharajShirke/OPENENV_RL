"""
reward.py — Gov Workflow OpenEnv Phase 4: Dense Reward Shaping

Formula (per step):
  R_t = progress_reward + completion_reward + recovery_reward + stability_bonus
        - waiting_penalty - sla_penalty - fairness_penalty
        - invalid_action_penalty - idle_capacity_penalty - oscillation_penalty

All coefficients are named constants — never magic numbers inline.
"""
from __future__ import annotations
from app.models import RewardModel

# ── Positive coefficients ─────────────────────────────────────────
COEFF_PROGRESS     = 0.7   # per stage advance
COEFF_COMPLETION   = 4.0   # per completed case
COEFF_RECOVERY     = 1.5   # per unblocked missing-doc case resolved
COEFF_STABILITY    = 0.1   # per step with zero SLA breaches and zero invalid actions

# ── Negative coefficients ─────────────────────────────────────────
COEFF_WAITING      = 0.04  # per case per day in backlog
COEFF_SLA          = 1.5   # per new SLA breach
COEFF_FAIRNESS     = 2.0   # per unit of fairness excess above threshold
COEFF_INVALID      = 1.5   # flat penalty per invalid action
COEFF_IDLE         = 0.05  # per idle officer-day
COEFF_OSCILLATION  = 0.15  # per oscillation event (repeated contradictory actions)

# ── Fairness default tolerance (when no threshold set by task) ────
DEFAULT_FAIRNESS_TOLERANCE = 0.40


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
    newly_unblocked_docs: int = 0,
    oscillation_detected: bool = False,
    award_stability_bonus: bool = True,
) -> RewardModel:
    """
    Compute one-step dense reward.

    Args:
        stage_advances:       Number of applications that moved forward one stage today.
        completions:          Number of applications fully completed today.
        active_backlog:       Total cases still pending (creates waiting pressure).
        new_sla_breaches:     New SLA deadline violations this step.
        fairness_gap:         Cross-service completion fairness gap [0.0, 1.0].
        fairness_threshold:   Task-defined acceptable fairness gap (or None → default).
        invalid_action:       Whether the submitted action was invalid.
        idle_capacity:        Officer-days wasted idle while backlog exists.
        newly_unblocked_docs: Cases unblocked after missing-doc resolution (positive signal).
        oscillation_detected: True if agent is rapidly reversing recent decisions.

    Returns:
        RewardModel with all components filled and total_reward as the scalar.
    """
    # ── Positive components ───────────────────────────────────────
    progress_reward   = COEFF_PROGRESS   * stage_advances
    completion_reward = COEFF_COMPLETION * completions
    recovery_reward   = COEFF_RECOVERY   * newly_unblocked_docs
    stability_bonus = (
        COEFF_STABILITY
        if (award_stability_bonus and new_sla_breaches == 0 and not invalid_action)
        else 0.0
    )

    # ── Negative components ───────────────────────────────────────
    waiting_penalty = COEFF_WAITING * active_backlog

    sla_penalty = COEFF_SLA * new_sla_breaches

    tolerance = fairness_threshold if fairness_threshold is not None else DEFAULT_FAIRNESS_TOLERANCE
    unfairness_excess = max(0.0, fairness_gap - tolerance)
    fairness_penalty = COEFF_FAIRNESS * unfairness_excess

    invalid_action_penalty = COEFF_INVALID if invalid_action else 0.0

    idle_capacity_penalty = COEFF_IDLE * idle_capacity

    oscillation_penalty = COEFF_OSCILLATION if oscillation_detected else 0.0

    # ── Total ─────────────────────────────────────────────────────
    total_reward = (
        progress_reward + completion_reward + recovery_reward + stability_bonus
        - waiting_penalty - sla_penalty - fairness_penalty
        - invalid_action_penalty - idle_capacity_penalty - oscillation_penalty
    )

    return RewardModel(
        total_reward=round(total_reward, 4),
        progress_reward=round(progress_reward, 4),
        completion_reward=round(completion_reward, 4),
        recovery_reward=round(recovery_reward, 4),
        stability_bonus=round(stability_bonus, 4),
        waiting_penalty=round(-waiting_penalty, 4),
        sla_penalty=round(-sla_penalty, 4),
        fairness_penalty=round(-fairness_penalty, 4),
        invalid_action_penalty=round(-invalid_action_penalty, 4),
        idle_capacity_penalty=round(-idle_capacity_penalty, 4),
        oscillation_penalty=round(-oscillation_penalty, 4),
    )
