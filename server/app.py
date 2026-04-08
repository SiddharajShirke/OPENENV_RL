from __future__ import annotations

import uvicorn

from app.config import server_settings
from app.main import app


def main() -> None:
    uvicorn.run(
        "app.main:app",
        host=server_settings.host,
        port=server_settings.port,
        log_level=server_settings.log_level,
        workers=1,
    )


if __name__ == "__main__":
    main()

