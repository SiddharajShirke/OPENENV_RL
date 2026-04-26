"""Tests for FeatureBuilder using the real ObservationModel schema."""

from __future__ import annotations

import numpy as np
import pytest

from app.models import (
    ObservationModel,
    OfficerPool,
    PriorityMode,
    QueueSnapshot,
    ServiceType,
    StageType,
)
from rl.feature_builder import FeatureBuilder, OBS_DIM


def _make_obs() -> ObservationModel:
    snapshots = {}
    for i, svc in enumerate(ServiceType):
        snapshots[svc] = QueueSnapshot(
            service_type=svc,
            public_stage_counts={
                StageType.SUBMISSION.value: 2 + i % 2,
                StageType.DOCUMENT_VERIFICATION.value: 1,
                StageType.FIELD_VERIFICATION.value: 1,
                StageType.APPROVAL.value: 0,
                StageType.ISSUANCE.value: 0,
            },
            total_pending=6 + i,
            blocked_missing_docs=i % 3,
            urgent_pending=2 if i % 3 == 0 else 1,
            total_sla_breached=0,
            avg_waiting_days=3.0 + i,
        )

    return ObservationModel(
        task_id="district_backlog_easy",
        episode_id="ep-test",
        day=8,
        max_days=20,
        officer_pool=OfficerPool(
            total_officers=len(ServiceType) + 2,
            available_officers=len(ServiceType) + 2,
            allocated={svc: 1 for svc in ServiceType},
        ),
        queue_snapshots=snapshots,
        total_backlog=sum(s.total_pending for s in snapshots.values()),
        total_completed=15,
        total_sla_breaches=3,
        fairness_index=1.0 - 0.12,
        escalation_budget_remaining=4,
        last_action_valid=True,
        last_action_message="ok",
    )


@pytest.fixture
def builder() -> FeatureBuilder:
    return FeatureBuilder()


def test_output_shape(builder: FeatureBuilder) -> None:
    assert builder.build(_make_obs()).shape == (OBS_DIM,)


def test_output_dtype(builder: FeatureBuilder) -> None:
    assert builder.build(_make_obs()).dtype == np.float32


def test_deterministic(builder: FeatureBuilder) -> None:
    obs = _make_obs()
    np.testing.assert_array_equal(
        builder.build(obs, "urgent_first", "advance_time"),
        builder.build(obs, "urgent_first", "advance_time"),
    )


def test_no_nan_or_inf(builder: FeatureBuilder) -> None:
    vec = builder.build(_make_obs())
    assert not np.any(np.isnan(vec))
    assert not np.any(np.isinf(vec))


def test_values_in_reasonable_range(builder: FeatureBuilder) -> None:
    vec = builder.build(_make_obs())
    assert np.all(vec >= 0.0)
    assert np.all(vec <= 1.0 + 1e-6)
