"""
graders.py — Gov Workflow OpenEnv: Deterministic Episode Graders

Rules:
  - All graders read ONLY from EpisodeStateModel flat fields.
  - No access to env internals, EpisodeMetrics, or reward breakdown proxies.
  - GraderResult uses the aligned schema (score, grader_name, named metric fields).
  - grade_episode() dispatches by task_id.

Grader weights:
  Easy   — completion(0.45) + SLA(0.35) + idle_efficiency(0.20)          = 1.00
  Medium — completion(0.35) + SLA(0.30) + doc_rework(0.20) + urgent(0.15) = 1.00
  Hard   — completion(0.28) + SLA(0.24) + doc_rework(0.16)
           + fairness(0.16) + escalation_discipline(0.16)                 = 1.00
"""
from __future__ import annotations
from app.models import EpisodeStateModel, GraderResult


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _safe_ratio(num: float, den: float, default: float = 1.0) -> float:
    """Safe division, clamped to [0.0, 1.0]. Returns `default` when den ≤ 0."""
    if den <= 0:
        return max(0.0, min(1.0, default))
    return max(0.0, min(1.0, num / den))


def _b(value: float) -> float:
    """Clamp any float to [0.0, 1.0]."""
    return max(0.0, min(1.0, float(value)))


def _extract(state: EpisodeStateModel) -> dict[str, float]:
    """
    Extract all grader input metrics from EpisodeStateModel flat fields.

    Design note:
      - total_arrived   : populated by env.state() from metrics.total_arrived
      - fairness_gap    : computed by completion_fairness_gap() in env.state()
      - All other fields are direct EpisodeStateModel attributes.
    """
    total_arrived      = max(1, state.total_arrived)
    total_completed    = float(state.total_completed)
    total_breaches     = float(state.total_sla_breaches)
    total_docs_req     = float(state.total_docs_requested)
    total_docs_cleared = float(state.total_docs_cleared)
    total_urgent_arr   = float(state.total_urgent_arrived)
    total_urgent_comp  = float(state.total_urgent_completed)
    total_idle         = float(state.total_idle_officer_days)
    total_capacity     = float(state.total_capacity_days)
    total_escused      = float(state.total_escalations_used)
    total_wasted_esc   = float(state.total_wasted_escalations)
    fairness_gap       = float(state.fairness_gap)

    return {
        "completion_rate":         _b(_safe_ratio(total_completed, total_arrived, 0.0)),
        "sla_compliance":          _b(1.0 - _safe_ratio(total_breaches, total_arrived, 0.0)),
        "document_rework_quality": _b(_safe_ratio(total_docs_cleared, total_docs_req, 1.0)),
        "urgent_served_rate":      _b(_safe_ratio(total_urgent_comp, total_urgent_arr, 1.0)),
        "fairness_score":          _b(1.0 - fairness_gap),
        "escalation_discipline":   _b(1.0 - _safe_ratio(total_wasted_esc, max(1.0, total_escused), 0.0)),
        "idle_efficiency":         _b(1.0 - _safe_ratio(total_idle, max(1.0, total_capacity), 0.0)),
        "fairness_gap":            round(fairness_gap, 4),
    }


def _build_result(
    state: EpisodeStateModel,
    score: float,
    grader_name: str,
    m: dict[str, float],
) -> GraderResult:
    """Assemble a fully-populated GraderResult from metric dict and state."""
    total_arrived = max(0, state.total_arrived)
    avg_wait = state.avg_waiting_days

    return GraderResult(
        task_id=state.task_id,
        episode_id=state.episode_id,
        grader_name=grader_name,
        score=_b(score),
        completion_rate=m["completion_rate"],
        sla_compliance_rate=m["sla_compliance"],
        idle_efficiency=m["idle_efficiency"],
        document_rework_quality=m["document_rework_quality"],
        urgent_served_rate=m["urgent_served_rate"],
        fairness_score=m["fairness_score"],
        escalation_discipline=m["escalation_discipline"],
        fairness_gap=m["fairness_gap"],
        total_cases_arrived=total_arrived,
        total_completed=state.total_completed,
        total_sla_breached=state.total_sla_breaches,
        total_rejected=state.total_rejected,
        avg_waiting_days=avg_wait,
    )


# ─────────────────────────────────────────────────────────────────────────────
# TASK GRADERS
# ─────────────────────────────────────────────────────────────────────────────

def grade_easy(state: EpisodeStateModel) -> GraderResult:
    """
    district_backlog_easy grader.
    Focus: raw throughput and SLA hygiene under simple single-service load.

    Weights: completion(0.45) + SLA(0.35) + idle_efficiency(0.20)
    """
    m = _extract(state)
    score = (
        0.45 * m["completion_rate"]
      + 0.35 * m["sla_compliance"]
      + 0.20 * m["idle_efficiency"]
    )
    return _build_result(state, score, "easy", m)


def grade_medium(state: EpisodeStateModel) -> GraderResult:
    """
    mixed_urgency_medium grader.
    Focus: throughput + SLA + document quality + prioritizing urgent cases.

    Weights: completion(0.35) + SLA(0.30) + doc_rework(0.20) + urgent(0.15)
    """
    m = _extract(state)
    score = (
        0.35 * m["completion_rate"]
      + 0.30 * m["sla_compliance"]
      + 0.20 * m["document_rework_quality"]
      + 0.15 * m["urgent_served_rate"]
    )
    return _build_result(state, score, "medium", m)


def grade_hard(state: EpisodeStateModel) -> GraderResult:
    """
    cross_department_hard grader.
    Focus: all-round excellence including cross-service fairness and
    restrained escalation use under crisis conditions.

    Weights: completion(0.28) + SLA(0.24) + doc_rework(0.16)
             + fairness(0.16) + escalation_discipline(0.16)
    """
    m = _extract(state)
    score = (
        0.28 * m["completion_rate"]
      + 0.24 * m["sla_compliance"]
      + 0.16 * m["document_rework_quality"]
      + 0.16 * m["fairness_score"]
      + 0.16 * m["escalation_discipline"]
    )
    return _build_result(state, score, "hard", m)


# ─────────────────────────────────────────────────────────────────────────────
# DISPATCHER
# ─────────────────────────────────────────────────────────────────────────────

_GRADER_MAP = {
    "district_backlog_easy":          grade_easy,
    "district_backlog_easy_extreme":  grade_easy,
    "mixed_urgency_medium":           grade_medium,
    "cross_department_hard":          grade_hard,
}


def grade_episode(state: EpisodeStateModel) -> GraderResult:
    """
    Dispatch to the correct task grader.
    Falls back to grade_hard for unknown task IDs (safe default for new tasks).
    """
    grader_fn = _GRADER_MAP.get(state.task_id, grade_hard)
    return grader_fn(state)