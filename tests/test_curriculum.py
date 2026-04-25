"""Tests for curriculum scheduler behavior."""

from __future__ import annotations

from app.tasks import TASKS
from rl.curriculum import (
    ALL_TASKS,
    TASK_EASY,
    TASK_HARD,
    TASK_MEDIUM,
    CurriculumScheduler,
)


def test_stage_transitions_at_correct_timesteps() -> None:
    sched = CurriculumScheduler(total_timesteps=1000, rng_seed=42)
    assert sched.current_stage(0) == 1
    assert sched.current_stage(299) == 1
    assert sched.current_stage(300) == 2
    assert sched.current_stage(699) == 2
    assert sched.current_stage(700) == 3
    assert sched.current_stage(999) == 3


def test_easy_only_in_stage_1() -> None:
    sched = CurriculumScheduler(total_timesteps=1000, rng_seed=42)
    for t in (0, 50, 150, 299):
        assert sched.sample_task(t) == TASK_EASY


def test_all_tasks_sampled_in_stage_3() -> None:
    sched = CurriculumScheduler(total_timesteps=1000, rng_seed=42)
    seen = set()
    for _ in range(500):
        seen.add(sched.sample_task(900))
    assert TASK_EASY in seen
    assert TASK_MEDIUM in seen
    assert TASK_HARD in seen
    assert seen.issubset(set(ALL_TASKS))


def test_deterministic_eval_seeds_never_change() -> None:
    assert TASKS["district_backlog_easy"].seed == 42
    assert TASKS["mixed_urgency_medium"].seed == 123
    assert TASKS["cross_department_hard"].seed == 999
