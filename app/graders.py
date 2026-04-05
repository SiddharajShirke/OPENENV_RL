from __future__ import annotations
from app.models import EpisodeStateModel, GraderResult
from app.utils import bounded_score


def _safe_ratio(num: float, den: float, default: float = 1.0) -> float:
    if den <= 0:
        return default
    return num / den


def _common_metrics(state: EpisodeStateModel) -> dict[str, float]:
    m = state.metrics
    return {
        "completion_rate":        bounded_score(_safe_ratio(m.total_completed, m.total_arrived, 0.0)),
        "sla_compliance":         bounded_score(1.0 - _safe_ratio(m.total_sla_breaches, m.total_arrived, 0.0)),
        "document_rework_quality":bounded_score(_safe_ratio(m.total_docs_cleared, m.total_docs_requested, 1.0)),
        "urgent_served_rate":     bounded_score(_safe_ratio(m.total_urgent_completed, m.total_urgent_arrived, 1.0)),
        "fairness_score":         bounded_score(1.0 - state.fairness_gap),
        "escalation_discipline":  bounded_score(1.0 - _safe_ratio(m.total_wasted_escalations, m.total_escalations_used, 0.0)),
        "idle_efficiency":        bounded_score(1.0 - _safe_ratio(m.total_idle_officer_days, m.total_capacity_days, 0.0)),
        "fairness_gap":           round(state.fairness_gap, 4),
    }


def grade_easy(state: EpisodeStateModel) -> GraderResult:
    m = _common_metrics(state)
    score = 0.45 * m["completion_rate"] + 0.35 * m["sla_compliance"] + 0.20 * m["idle_efficiency"]
    return GraderResult(score=bounded_score(score), metrics=m, grader_name="easy")


def grade_medium(state: EpisodeStateModel) -> GraderResult:
    m = _common_metrics(state)
    score = (0.35 * m["completion_rate"] + 0.30 * m["sla_compliance"]
           + 0.20 * m["document_rework_quality"] + 0.15 * m["urgent_served_rate"])
    return GraderResult(score=bounded_score(score), metrics=m, grader_name="medium")


def grade_hard(state: EpisodeStateModel) -> GraderResult:
    m = _common_metrics(state)
    score = (0.28 * m["completion_rate"] + 0.24 * m["sla_compliance"]
           + 0.16 * m["document_rework_quality"] + 0.16 * m["fairness_score"]
           + 0.16 * m["escalation_discipline"])
    return GraderResult(score=bounded_score(score), metrics=m, grader_name="hard")


def grade_episode(state: EpisodeStateModel) -> GraderResult:
    if state.task_id == "district_backlog_easy":
        return grade_easy(state)
    if state.task_id == "mixed_urgency_medium":
        return grade_medium(state)
    return grade_hard(state)