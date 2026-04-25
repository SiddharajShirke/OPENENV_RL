"""
Converts ObservationModel (Pydantic) → flat numpy float32 vector.
All downstream RL code depends on OBS_DIM being stable.

Feature layout (total = OBS_DIM = 84):
  [0  : 63) — per-service block  (7 services × 9 features each)
  [63 : 84) — global block       (21 scalar features)
"""

from __future__ import annotations

import numpy as np
from typing import List

from app.models import (
    ObservationModel,
    ServiceType,
    StageType,
    PriorityMode,
    ActionType,
)

# ── Canonical orderings (must never change across the codebase) ──────────────
SERVICES: List[ServiceType] = list(ServiceType)          # 7 services
STAGES:   List[StageType]   = list(StageType)            # 5 stages
PRIORITY_MODES: List[PriorityMode] = list(PriorityMode)  # 4 modes
ACTION_TYPES:   List[ActionType]   = list(ActionType)    # 6 types

SERVICE_IDX = {s: i for i, s in enumerate(SERVICES)}
STAGE_IDX   = {s: i for i, s in enumerate(STAGES)}
PM_IDX      = {m: i for i, m in enumerate(PRIORITY_MODES)}
AT_IDX      = {a: i for i, a in enumerate(ACTION_TYPES)}

# ── Dimension constants ───────────────────────────────────────────────────────
N_SERVICES      = len(SERVICES)   # 7
N_STAGES        = len(STAGES)     # 5
N_PRIORITY_MODES = len(PRIORITY_MODES)  # 4
N_ACTION_TYPES  = len(ACTION_TYPES)     # 6

PER_SERVICE_DIM = 4 + N_STAGES           # queue_len, avg_wait, urgent, missing + 5 stage fracs = 9
GLOBAL_DIM = (
    1                   # day_ratio
    + 1                 # total_backlog_normalized
    + 1                 # total_completed_normalized
    + 1                 # total_sla_breaches_normalized
    + 1                 # fairness_gap
    + 1                 # escalation_budget_ratio
    + 1                 # last_action_valid
    + N_ACTION_TYPES    # last_action_type one-hot  (6)
    + N_PRIORITY_MODES  # current_priority_mode one-hot (4)
    + 1                 # idle_officer_ratio
    + 1                 # urgent_backlog_ratio
    + 1                 # officer_utilization
    + 1                 # backlog_per_officer
)  # = 21

OBS_DIM = N_SERVICES * PER_SERVICE_DIM + GLOBAL_DIM  # 63 + 21 = 84

# ── Normalisation caps (avoid div-by-zero, keep values in [0,1]) ─────────────
_MAX_QUEUE      = 200.0
_MAX_WAIT       = 30.0
_MAX_URGENT     = 50.0
_MAX_MISSING    = 50.0
_MAX_BACKLOG    = 500.0
_MAX_COMPLETED  = 500.0
_MAX_SLA        = 100.0
_MAX_ESC_BUDGET = 20.0
_MAX_OFFICERS   = 50.0


class FeatureBuilder:
    """
    Stateless transformer: ObservationModel → np.ndarray[float32, OBS_DIM].

    Usage:
        fb = FeatureBuilder()
        vec = fb.build(obs, current_priority_mode="urgent_first",
                       last_action_type="advance_time")
    """

    def build(
        self,
        obs: ObservationModel,
        current_priority_mode: str = "balanced",
        last_action_type: str = "advance_time",
    ) -> np.ndarray:
        features = np.zeros(OBS_DIM, dtype=np.float32)
        offset = 0

        snap_dict = {
            snap.service_type: snap 
            for snap in (obs.queue_snapshots.values() if isinstance(obs.queue_snapshots, dict) else obs.queue_snapshots)
        }

        # ── Per-service block ─────────────────────────────────────────────
        for svc in SERVICES:
            snap = snap_dict.get(svc)
            if snap is None:
                offset += PER_SERVICE_DIM
                continue

            total_in_svc = max(getattr(snap, "active_cases", getattr(snap, "total_pending", 0)), 1) # instead of queue_length

            features[offset + 0] = getattr(snap, "active_cases", getattr(snap, "total_pending", 0)) / _MAX_QUEUE
            features[offset + 1] = getattr(snap, "avg_age_days", getattr(snap, "avg_waiting_days", getattr(snap, "oldest_case_age_days", 0))) / _MAX_WAIT
            features[offset + 2] = getattr(snap, "urgent_cases", getattr(snap, "urgent_pending", 0)) / _MAX_URGENT
            features[offset + 3] = getattr(snap, "missing_docs_cases", getattr(snap, "blocked_missing_docs", 0)) / _MAX_MISSING

            # Stage distribution as fractions
            stage_counts = getattr(snap, "stage_counts", getattr(snap, "public_stage_counts", {})) or {}
            for stg in STAGES:
                count = stage_counts.get(stg, 0)
                features[offset + 4 + STAGE_IDX[stg]] = count / total_in_svc

            offset += PER_SERVICE_DIM

        # ── Global block ──────────────────────────────────────────────────
        day_ratio = obs.day / max(obs.max_days, 1)
        features[offset + 0]  = day_ratio
        features[offset + 1]  = obs.total_backlog    / _MAX_BACKLOG
        features[offset + 2]  = obs.total_completed  / _MAX_COMPLETED
        features[offset + 3]  = obs.total_sla_breaches / _MAX_SLA
        features[offset + 4]  = float(getattr(obs, "fairness_gap", getattr(obs, "fairness_index", 0.0)) or 0.0)
        features[offset + 5]  = obs.escalation_budget_remaining / _MAX_ESC_BUDGET
        features[offset + 6]  = float(obs.last_action_valid)
        offset += 7

        # Last action type one-hot
        at_vec = np.zeros(N_ACTION_TYPES, dtype=np.float32)
        try:
            at_vec[AT_IDX[ActionType(last_action_type)]] = 1.0
        except (ValueError, KeyError):
            pass
        features[offset: offset + N_ACTION_TYPES] = at_vec
        offset += N_ACTION_TYPES

        # Current priority mode one-hot
        pm_vec = np.zeros(N_PRIORITY_MODES, dtype=np.float32)
        try:
            pm_vec[PM_IDX[PriorityMode(current_priority_mode)]] = 1.0
        except (ValueError, KeyError):
            pass
        features[offset: offset + N_PRIORITY_MODES] = pm_vec
        offset += N_PRIORITY_MODES

        # Officer-derived scalars
        pool = obs.officer_pool
        total_officers = max(getattr(pool, "total_officers", 1) if not callable(getattr(pool, "total_officers", None)) else pool.total_officers(), 1)
        idle_officers = getattr(pool, "idle_officers", getattr(pool, "reserve_officers", 0))
        idle_ratio         = idle_officers / total_officers
        total_backlog_safe = max(obs.total_backlog, 1)
        urgent_total       = sum(
            getattr(snap_dict[s], "urgent_cases", getattr(snap_dict[s], "urgent_pending", 0))
            for s in SERVICES
            if s in snap_dict
        )
        urgent_ratio       = urgent_total / total_backlog_safe
        utilization        = (total_officers - idle_officers) / total_officers
        backlog_per_off    = obs.total_backlog / total_officers

        features[offset + 0] = float(np.clip(idle_ratio,      0.0, 1.0))
        features[offset + 1] = float(np.clip(urgent_ratio,    0.0, 1.0))
        features[offset + 2] = float(np.clip(utilization,     0.0, 1.0))
        features[offset + 3] = float(np.clip(backlog_per_off / _MAX_OFFICERS, 0.0, 1.0))

        assert offset + 4 == OBS_DIM, f"OBS_DIM mismatch: {offset + 4} != {OBS_DIM}"
        return features


# -- Action space layout (N_ACTIONS = 28) -------------------------------------
#
#  0 - 3  : set_priority_mode (4 modes in PRIORITY_MODES order)
#  4 - 10 : request_missing_documents per service (7)
# 11 - 17 : escalate_service per service (7)
# 18      : advance_time
# 19 - 25 : reallocate_officers from source service -> most loaded other service
# 26      : assign_capacity +1 to most-loaded service
# 27      : assign_capacity +1 to most-urgent service

N_ACTIONS = 4 + N_SERVICES + N_SERVICES + 1 + N_SERVICES + 2  # = 28

ACTION_DECODE_TABLE = {}
idx = 0
for m in PRIORITY_MODES:
    ACTION_DECODE_TABLE[idx] = ("set_priority_mode", None, m.value, None)
    idx += 1
for s in SERVICES:
    ACTION_DECODE_TABLE[idx] = ("request_missing_documents", s.value, None, None)
    idx += 1
for s in SERVICES:
    ACTION_DECODE_TABLE[idx] = ("escalate_service", s.value, None, None)
    idx += 1
ACTION_DECODE_TABLE[idx] = ("advance_time", None, None, None); idx += 1
for s in SERVICES:
    ACTION_DECODE_TABLE[idx] = ("reallocate_officers", s.value, "most_loaded_other", 1)
    idx += 1
ACTION_DECODE_TABLE[idx] = ("assign_capacity", "__most_loaded__", None, 1); idx += 1
ACTION_DECODE_TABLE[idx] = ("assign_capacity", "__most_urgent__", None, 1); idx += 1

assert len(ACTION_DECODE_TABLE) == N_ACTIONS
