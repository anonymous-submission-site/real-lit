"""Structured per-run event emission via JSONL.

POSIX guarantees sub-PIPE_BUF (4 KB on Linux) `O_APPEND | write()` is atomic for
concurrent writers; each event is a single JSON line ≤ 1 KB, so multi-process
appends from concurrent workers are safe without locks.
"""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def emit_event(path: str | Path, event: str, **fields: Any) -> None:
    """Append a structured event line to `path`."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z"),
        "event": event,
        **fields,
    }
    line = json.dumps(record, default=str) + "\n"
    with open(p, "ab") as f:
        f.write(line.encode())


def progress_path_for_run(logs_dir: str | Path, run_name: str) -> Path:
    return Path(logs_dir) / "runs" / run_name / "progress.jsonl"
