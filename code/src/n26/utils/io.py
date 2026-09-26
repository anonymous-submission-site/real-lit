"""I/O helpers."""
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any


def atomic_json_write(path: str | Path, obj: Any, *, indent: int = 2) -> None:
    """Atomically write obj as JSON to path.

    Writes to <path>.tmp then renames. POSIX `os.replace` guarantees the
    target file is never partially written from a reader's perspective.
    On serialization failure, the .tmp file is removed.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    try:
        tmp.write_text(json.dumps(obj, indent=indent, default=str))
        os.replace(tmp, p)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
