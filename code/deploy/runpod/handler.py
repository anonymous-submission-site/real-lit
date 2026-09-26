"""RunPod Serverless handler — runs one RealLit benchmark cell per invocation.

Each invocation:
  1. Receives ``{"input": {"experiment_id": int, "run_name": str}}``.
  2. Looks the cell up in ``docs/ablation_matrix.csv``.
  3. Runs it via ``scripts/run_cell.py``.
  4. Writes the result JSON to the Network Volume at
     ``/runpod-volume/results/<run_name>/<experiment_id>.json``.
  5. Returns a small status payload (the full cell record is on the volume,
     pulled later by ``collect_results.py``).
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from scripts.run_cell import main as run_cell_main  # noqa: E402

LOG = logging.getLogger("n26.runpod.handler")
DEFAULT_MANIFEST = os.environ.get("N26_MANIFEST", "docs/ablation_matrix.csv")
DEFAULT_RESULTS_ROOT = os.environ.get("N26_RESULTS_ROOT", "/runpod-volume/results")


def handler(event: dict) -> dict:
    inp = event.get("input") or {}
    try:
        experiment_id = int(inp["experiment_id"])
        run_name = str(inp["run_name"])
    except (KeyError, TypeError, ValueError) as exc:
        return {
            "status": f"error: bad input: {exc}",
            "rc": None,
            "experiment_id": inp.get("experiment_id"),
        }

    manifest = inp.get("manifest", DEFAULT_MANIFEST)
    out_dir = Path(inp.get("out_dir", f"{DEFAULT_RESULTS_ROOT}/{run_name}"))

    argv = [
        "--manifest", str(manifest),
        "--experiment-id", str(experiment_id),
        "--out-dir", str(out_dir),
        "--overwrite",
    ]
    LOG.info("dispatching cell experiment_id=%s run=%s", experiment_id, run_name)
    try:
        rc = run_cell_main(argv)
    except Exception as exc:
        LOG.exception("run_cell.main raised")
        return {
            "status": f"error: {type(exc).__name__}: {exc}",
            "rc": None,
            "experiment_id": experiment_id,
        }

    cell_path = out_dir / f"{experiment_id}.json"
    return {
        "status": "ok" if rc == 0 else f"rc={rc}",
        "rc": rc,
        "experiment_id": experiment_id,
        "volume_path": (
            str(cell_path.relative_to("/runpod-volume"))
            if cell_path.exists() and str(cell_path).startswith("/runpod-volume/")
            else (str(cell_path) if cell_path.exists() else None)
        ),
    }


if __name__ == "__main__":
    import runpod  # type: ignore
    runpod.serverless.start({"handler": handler})
