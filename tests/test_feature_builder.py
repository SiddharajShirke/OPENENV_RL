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
    snapshots = []
    for i, svc in enumerate(ServiceType):
        snapshots.append(
            QueueSnapshot(
                service=svc,
                stage_counts={
                    StageType.SUBMISSION: 2 + i % 2,
                    StageType.DOCUMENT_VERIFICATION: 1,
                    StageType.FIELD_VERIFICATION: 1,
                    StageType.APPROVAL: 0,
                    StageType.ISSUANCE: 0,
                },
                active_cases=6 + i,
                missing_docs_cases=i % 3,
                escalated_cases=1 if i % 2 else 0,
                urgent_cases=2 if i % 3 == 0 else 1,
                breached_cases=0,
                avg_age_days=3.0 + i,
            )
        )

    return ObservationModel(
        task_id="district_backlog_easy",
        day=8,
        max_days=20,
        priority_mode=PriorityMode.BALANCED,
        officer_pool=OfficerPool(
            allocations={svc: 1 for svc in ServiceType},
            reserve_officers=2,
        ),
        queue_snapshots=snapshots,
        total_backlog=sum(s.active_cases for s in snapshots),
        total_completed=15,
        total_sla_breaches=3,
        fairness_gap=0.12,
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
