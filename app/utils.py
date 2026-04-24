"""
utils.py — Shared pure-function helpers.
No imports from env.py or simulator.py (prevents circular imports).
"""
from __future__ import annotations
from app.models import ServiceType


def completion_fairness_gap(
    arrived_by_service: dict,
    completed_by_service: dict,
) -> float:
    """
    Fairness gap = max completion rate difference across services.
    Returns 0.0 if only one service, 1.0 if perfectly unfair.
    """
    rates = []
    for svc in arrived_by_service:
        arrived   = arrived_by_service.get(svc, 0)
        completed = completed_by_service.get(svc, 0)
        if arrived > 0:
            rates.append(completed / arrived)
    if len(rates) < 2:
        return 0.0
    return round(max(rates) - min(rates), 4)
