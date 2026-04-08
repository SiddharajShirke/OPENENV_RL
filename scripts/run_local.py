#!/usr/bin/env python3
"""
Run the Gov Workflow OpenEnv FastAPI app locally.

Usage:
    python scripts/run_local.py
    python scripts/run_local.py --host 0.0.0.0 --port 7860 --reload
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import uvicorn

# Ensure project root is importable when script is executed directly.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.config import server_settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local OpenEnv FastAPI server")
    parser.add_argument("--host", default=server_settings.host)
    parser.add_argument("--port", type=int, default=server_settings.port)
    parser.add_argument("--log-level", default=server_settings.log_level)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        workers=1,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
