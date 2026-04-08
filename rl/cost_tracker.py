"""
Separates RewardModel fields into reward signal r_t and cost signals c_t.
Phase 1-3 : costs logged only (diagnostic).
Phase 4   : costs drive Lagrangian multiplier updates.

Thresholds:
  d_sla        = 0.15  (max 15% SLA breach rate)
  d_fairness   = 0.20  (max 0.20 fairness gap)
  d_escalation = 0.10  (max 10% wasted escalation)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List
import numpy as np

THRESHOLD_SLA        = 0.15
THRESHOLD_FAIRNESS   = 0.20
THRESHOLD_ESCALATION = 0.10


@dataclass
class CostRecord:
    step:         int
    c_sla:        float
    c_fairness:   float
    c_escalation: float
    c_invalid:    float
    c_idle:       float


@dataclass
class EpisodeCostSummary:
    mean_c_sla:          float
    mean_c_fairness:     float
    mean_c_escalation:   float
    sla_violated:        bool
    fairness_violated:   bool
    escalation_violated: bool
    total_steps:         int


class CostTracker:
    """Accumulates per-step cost signals across an episode."""

    def __init__(self) -> None:
        self._records: List[CostRecord] = []
        self._step = 0

    def reset(self) -> None:
        self._records.clear()
        self._step = 0

    def record(self, reward_breakdown: dict) -> CostRecord:
        rec = CostRecord(
            step=self._step,
            c_sla=abs(float(reward_breakdown.get("sla_penalty",          0.0))),
            c_fairness=abs(float(reward_breakdown.get("fairness_penalty", 0.0))),
            c_escalation=abs(float(reward_breakdown.get("invalid_action_penalty", 0.0))),
            c_invalid=abs(float(reward_breakdown.get("invalid_action_penalty",    0.0))),
            c_idle=abs(float(reward_breakdown.get("idle_capacity_penalty",        0.0))),
        )
        self._records.append(rec)
        self._step += 1
        return rec

    def summarise(self) -> EpisodeCostSummary:
        if not self._records:
            return EpisodeCostSummary(0.0, 0.0, 0.0, False, False, False, 0)
        mean_sla  = float(np.mean([r.c_sla        for r in self._records]))
        mean_fair = float(np.mean([r.c_fairness   for r in self._records]))
        mean_esc  = float(np.mean([r.c_escalation for r in self._records]))
        return EpisodeCostSummary(
            mean_c_sla=mean_sla,
            mean_c_fairness=mean_fair,
            mean_c_escalation=mean_esc,
            sla_violated=(mean_sla  > THRESHOLD_SLA),
            fairness_violated=(mean_fair > THRESHOLD_FAIRNESS),
            escalation_violated=(mean_esc  > THRESHOLD_ESCALATION),
            total_steps=len(self._records),
        )
