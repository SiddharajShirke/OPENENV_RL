from __future__ import annotations

import uvicorn
from openenv.core import create_app

from server.gov_environment import (
    GovWorkflowAction,
    GovWorkflowObservation,
    GovWorkflowOpenEnv,
)


def _env_factory() -> GovWorkflowOpenEnv:
    return GovWorkflowOpenEnv(task_id="district_backlog_easy", seed=42)


app = create_app(
    env=_env_factory,
    action_cls=GovWorkflowAction,
    observation_cls=GovWorkflowObservation,
    env_name="gov-workflow-openenv",
)


def main() -> None:
    uvicorn.run("server.app:app", host="0.0.0.0", port=7861, log_level="info")


if __name__ == "__main__":
    main()

