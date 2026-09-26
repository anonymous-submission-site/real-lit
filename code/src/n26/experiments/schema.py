"""Per-cell JSON output schema (v1.0.0)."""
from __future__ import annotations

SCHEMA_VERSION = "1.0.0"


def make_initial_record(
    *,
    experiment_id: int | None,
    run_name: str,
    cell: dict,
) -> dict:
    """Build the starting record for a cell (status='running')."""
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "run_name": run_name,
        "status": "running",
        "cell": cell,
        "data_shape": None,
        "metrics": {"imputation": None, "classification": None},
        "subgroup_metrics": {},
        "timing": {},
        "provenance": _provenance(),
        "config": {},
        "errors": None,
        "warnings": [],
    }


def _provenance() -> dict:
    """Capture environment for reproducibility."""
    import sys
    import platform

    return {
        "git_sha": None,
        "git_dirty": None,
        "python_version": sys.version.split()[0],
        "package_versions": _pkg_versions(),
        "host": platform.node(),
        "gpu_name": _gpu_name(),
        "cuda": _cuda_version(),
    }


def _pkg_versions() -> dict[str, str]:
    import importlib.metadata as md

    pkgs = ["numpy", "pandas", "scikit-learn", "scipy", "torch", "tabpfn"]
    out = {}
    for p in pkgs:
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            pass
    return out


def _gpu_name() -> str | None:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return None


def _cuda_version() -> str | None:
    try:
        import torch

        return torch.version.cuda
    except Exception:
        return None
