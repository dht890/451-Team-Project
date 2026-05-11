"""
Run the API with auto-reload, excluding venv and uploads from the file watcher.

Uvicorn's WatchFiles filter matches exclude directories against absolute paths;
relative `--reload-exclude venv` often fails on Windows, so tools touch venv and
trigger endless reloads (especially under OneDrive).

Usage (from project root, venv activated):
  python dev_server.py
"""

from __future__ import annotations

from pathlib import Path

import uvicorn

_ROOT = Path(__file__).resolve().parent

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        reload_excludes=[
            str(_ROOT / "venv"),
            str(_ROOT / "uploads"),
        ],
    )
