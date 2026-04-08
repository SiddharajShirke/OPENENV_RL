"""
Curriculum scheduler for staged training.

Stage 1 (0-30%)  : Easy only
Stage 2 (30-70%) : Easy + Medium (50/50)
Stage 3 (70-100%): All 3 tasks (20/40/40 weights)
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Tuple

TASK_EASY   = "district_backlog_easy"
TASK_MEDIUM = "mixed_urgency_medium"
TASK_HARD   = "cross_department_hard"
ALL_TASKS   = [TASK_EASY, TASK_MEDIUM, TASK_HARD]


@dataclass
class CurriculumConfig:
    stage1_end_frac: float = 0.30
    stage2_end_frac: float = 0.70
    stage3_weights: Tuple[float, ...] = (0.20, 0.40, 0.40)


class CurriculumScheduler:
    """
    Selects task_id for next training episode based on training progress.
    """

    def __init__(
        self,
        total_timesteps: int,
        config: CurriculumConfig | None = None,
        rng_seed: int = 0,
    ):
        self.total_timesteps = total_timesteps
        self.cfg  = config or CurriculumConfig()
        self._rng = random.Random(rng_seed)

    def sample_task(self, current_timestep: int) -> str:
        progress = current_timestep / max(self.total_timesteps, 1)
        if progress < self.cfg.stage1_end_frac:
            return TASK_EASY
        elif progress < self.cfg.stage2_end_frac:
            return self._rng.choice([TASK_EASY, TASK_MEDIUM])
        else:
            return self._rng.choices(ALL_TASKS, weights=list(self.cfg.stage3_weights), k=1)[0]

    def current_stage(self, current_timestep: int) -> int:
        progress = current_timestep / max(self.total_timesteps, 1)
        if progress < self.cfg.stage1_end_frac:
            return 1
        elif progress < self.cfg.stage2_end_frac:
            return 2
        return 3
