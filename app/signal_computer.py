"""
signal_computer.py — Gov Workflow OpenEnv v2.0
Computes normalized compressed state signals for observations.
All signals are deterministic and normalized to [0.0, 1.0].
"""
from typing import Dict
from app.models import QueueSnapshot, OfficerPool


class ComputedSignals:
    def __init__(self):
        self.backlog_pressure: float = 0.0
        self.sla_risk_score: float = 0.0
        self.fairness_index: float = 1.0
        self.resource_utilization: float = 0.0
        self.digital_intake_ratio: float = 0.5
        self.blocked_cases_missing_docs: int = 0
        self.blocked_cases_enrichment: int = 0
        self.field_verification_load: float = 0.0


class SignalComputer:
    def compute(
        self,
        queue_snapshots: Dict[str, QueueSnapshot],
        officer_pool: OfficerPool,
        todays_arrivals: int = 0,
        digital_arrivals: int = 0,
        capacity_per_day: float = 1.0,
    ) -> ComputedSignals:
        signals = ComputedSignals()
        snapshots = list(queue_snapshots.values())
        if not snapshots:
            return signals

        total_pending = sum(s.total_pending for s in snapshots)

        # Backlog pressure
        capacity_ceiling = max(1.0, capacity_per_day * 5.0)
        signals.backlog_pressure = min(1.0, total_pending / capacity_ceiling)

        # SLA risk score (weighted average)
        total_nonzero = max(1, total_pending)
        signals.sla_risk_score = min(1.0, max(0.0,
            sum(s.current_sla_risk * s.total_pending for s in snapshots) / total_nonzero
        ))

        # Fairness index (1 - coefficient of variation of completion rates)
        if len(snapshots) < 2:
            signals.fairness_index = 1.0
        else:
            rates = []
            for s in snapshots:
                total = s.total_pending + s.total_completed_today
                rates.append(s.total_completed_today / max(1, total) if total > 0 else 0.0)
            mean = sum(rates) / len(rates)
            if mean > 0:
                variance = sum((r - mean) ** 2 for r in rates) / len(rates)
                cv = (variance ** 0.5) / mean
                signals.fairness_index = max(0.0, 1.0 - min(1.0, cv))
            else:
                signals.fairness_index = 1.0

        # Resource utilization
        allocated = sum(officer_pool.allocated.values())
        signals.resource_utilization = min(1.0, allocated / max(1, officer_pool.available_officers))

        # Digital intake ratio
        signals.digital_intake_ratio = (
            min(1.0, digital_arrivals / todays_arrivals) if todays_arrivals > 0 else 0.5
        )

        # Blocked cases
        signals.blocked_cases_missing_docs = sum(s.blocked_missing_docs for s in snapshots)
        signals.blocked_cases_enrichment   = sum(s.blocked_enrichment for s in snapshots)

        # Field verification load
        total_in_field = sum(s.field_verification_pending for s in snapshots)
        signals.field_verification_load = total_in_field / total_nonzero if total_nonzero > 0 else 0.0

        return signals
