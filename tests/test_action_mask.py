"""Tests for ActionMaskComputer -- pure logic, no env dependency."""

import numpy as np
import pytest
from types import SimpleNamespace

from rl.action_mask import ActionMaskComputer
from rl.feature_builder import ACTION_DECODE_TABLE, N_ACTIONS
from app.models import ServiceType


def _make_obs(
    escalation_budget=5,
    missing_doc_counts=None,
    urgent_counts=None,
    reserve_officers=3,
    allocations=None,
    active_cases_by_service=None,
):
    services           = [s for s in ServiceType]
    missing_doc_counts = missing_doc_counts or {}
    urgent_counts      = urgent_counts or {}
    active_cases_by_service = active_cases_by_service or {svc.value: 10 for svc in services}
    allocations = allocations or {svc: 1 for svc in services}
    snapshots = {
        svc.value: SimpleNamespace(
            service_type=svc,
            total_pending=active_cases_by_service.get(svc.value, 0),
            avg_waiting_days=3.0,
            urgent_pending=urgent_counts.get(svc.value, 2),
            blocked_missing_docs=missing_doc_counts.get(svc.value, 0),
            escalated_cases=0,
            public_stage_counts={},
        )
        for svc in services
    }
    return SimpleNamespace(
        queue_snapshots=snapshots,
        escalation_budget_remaining=escalation_budget,
        officer_pool=SimpleNamespace(
            total_officers=lambda: 10,
            allocated=allocations,
            idle_officers=reserve_officers,
        ),
        day=5, max_days=30, total_backlog=50, total_completed=20,
        total_sla_breaches=3, fairness_gap=0.1,
        last_action_valid=True, last_action_message="ok",
    )


@pytest.fixture
def amc():
    return ActionMaskComputer()


def test_advance_time_always_valid(amc):
    assert amc.compute(_make_obs(), "balanced")[18]


def test_escalate_blocked_when_budget_zero(amc):
    mask = amc.compute(_make_obs(escalation_budget=0, urgent_counts={"passport": 5}), "balanced")
    for idx, (t, _, _, _) in ACTION_DECODE_TABLE.items():
        if t == "escalate_service":
            assert not mask[idx]


def test_missing_docs_blocked_when_no_pending(amc):
    mask = amc.compute(_make_obs(missing_doc_counts={}), "balanced")
    for idx, (t, _, _, _) in ACTION_DECODE_TABLE.items():
        if t == "request_missing_documents":
            assert not mask[idx]


def test_missing_docs_valid_when_pending(amc):
    first_svc = list(ServiceType)[0].value
    mask = amc.compute(_make_obs(missing_doc_counts={first_svc: 3}), "balanced")
    for idx, (t, s, _, _) in ACTION_DECODE_TABLE.items():
        if t == "request_missing_documents" and s == first_svc:
            assert mask[idx]


def test_reallocate_blocked_when_source_has_no_alloc(amc):
    zero_alloc = {svc: 0 for svc in ServiceType}
    mask = amc.compute(_make_obs(allocations=zero_alloc), "balanced")
    for idx, (t, _, _, _) in ACTION_DECODE_TABLE.items():
        if t == "reallocate_officers":
            assert not mask[idx]


def test_assign_capacity_blocked_when_no_reserve(amc):
    mask = amc.compute(_make_obs(reserve_officers=0), "balanced")
    for idx, (t, _, _, _) in ACTION_DECODE_TABLE.items():
        if t == "assign_capacity":
            assert not mask[idx]


def test_reallocate_blocked_when_only_one_active_service(amc):
    first = list(ServiceType)[0].value
    active_cases = {svc.value: 0 for svc in ServiceType}
    active_cases[first] = 10
    mask = amc.compute(_make_obs(active_cases_by_service=active_cases), "balanced")
    for idx, (t, _, _, _) in ACTION_DECODE_TABLE.items():
        if t == "reallocate_officers":
            assert not mask[idx]


def test_redundant_priority_mode_blocked(amc):
    mask = amc.compute(_make_obs(), current_priority_mode="urgent_first")
    assert not mask[0]


def test_mask_length(amc):
    assert len(amc.compute(_make_obs(), "balanced")) == N_ACTIONS


def test_at_least_one_valid_action(amc):
    assert amc.compute(_make_obs(), "balanced").any()
