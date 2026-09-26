#!/usr/bin/env python3
"""Pool per-fold JSONs into a per-cell ablation_results.csv.

Walks the per-fold tree written by ``scripts/run_cell.py``::

    <results-dir>/<run_name>/<dataset>/<target>/<regime>/<rate>/<imputer>/<classifier>/seed<N>.json

For every cell coordinate it collects the seed/fold records and
computes:

  - ``auc_mean``, ``auc_std``         test ROC-AUC, n=#folds
  - ``acc_mean``, ``acc_std``         test accuracy
  - ``rmse_mean``, ``rmse_std``       imputation RMSE on artificially-masked cells
  - ``mae_mean``, ``mae_std``         imputation MAE on artificially-masked cells
  - ``mmd_mean``, ``mmd_std``         imputation MMD (NaN-aware)
  - ``runtime_sec_total``             sum of fold-level total_sec
  - ``n_folds``                        how many seed JSONs were found
  - ``status``                         "completed" if all folds ok, else "partial" or "error"

Optionally merges these per-cell rows back into the manifest at
``--manifest`` (default ``docs/ablation_matrix.csv``), filling in the
``auc_mean``, ``acc_mean``, ``runtime_sec``, ``status`` columns for
the matching ``experiment_id``s.

Usage::

    # Per-cell summary only:
    python scripts/aggregate.py \\
        --results-dir results --run-name dev_smoke \\
        --out results/dev_smoke/ablation_results.csv

    # Plus merge into manifest:
    python scripts/aggregate.py \\
        --results-dir results --run-name dev_smoke \\
        --manifest docs/ablation_matrix.csv \\
        --merge-into results/dev_smoke/manifest_filled.csv
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Keys we pool seeds across.
_CELL_KEY = ("dataset", "target", "regime", "rate", "imputer", "classifier", "K")


def _safe_load(p: Path) -> dict | None:
    """Load a per-fold JSON. NaN tokens are mapped to math.nan."""
    try:
        text = p.read_text()
        return json.loads(text, parse_constant=lambda c: math.nan)
    except json.JSONDecodeError as exc:
        print(f"[warn] {p}: {exc}", file=sys.stderr)
        return None


def _walk_run(results_dir: Path, run_name: str | None) -> list[dict]:
    """Collect every seed*.json under results_dir (optionally scoped to run_name)."""
    base = results_dir / run_name if run_name else results_dir
    if not base.exists():
        raise SystemExit(f"results dir not found: {base}")
    out: list[dict] = []
    for p in sorted(base.rglob("seed*.json")):
        rec = _safe_load(p)
        if rec is None:
            continue
        rec.setdefault("_path", str(p))
        out.append(rec)
    return out


def _pool_cells(records: list[dict]) -> pd.DataFrame:
    """Group per-fold records by cell coordinates; aggregate metrics."""
    rows: dict[tuple, dict] = {}
    for rec in records:
        cell = rec.get("cell") or {}
        K = (rec.get("config") or {}).get("K", 1)
        key = tuple([cell.get(k, K if k == "K" else None) for k in _CELL_KEY])
        bucket = rows.setdefault(key, {
            "experiment_ids": [],
            "aucs": [], "accs": [],
            "auc_trains": [], "acc_trains": [],
            "rmses": [], "maes": [], "mmds": [],
            "runtime_secs": [],
            "statuses": [],
            "seeds": [],
        })
        bucket["experiment_ids"].append(rec.get("experiment_id"))
        bucket["statuses"].append(rec.get("status"))
        bucket["seeds"].append(cell.get("seed"))
        clf = (rec.get("metrics") or {}).get("classification") or {}
        bucket["aucs"].append(clf.get("auc"))
        bucket["accs"].append(clf.get("acc"))
        bucket["auc_trains"].append(clf.get("auc_train"))
        bucket["acc_trains"].append(clf.get("acc_train"))
        imp = (rec.get("metrics") or {}).get("imputation") or {}
        bucket["rmses"].append(imp.get("rmse_test"))
        bucket["maes"].append(imp.get("mae_test"))
        bucket["mmds"].append(imp.get("mmd_test"))
        timing = rec.get("timing") or {}
        bucket["runtime_secs"].append(timing.get("total_sec"))

    def _stat(values):
        arr = np.asarray([v for v in values if v is not None], dtype=float)
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            return float("nan"), float("nan")
        return float(arr.mean()), float(arr.std(ddof=1)) if arr.size > 1 else 0.0

    pooled = []
    for key, b in rows.items():
        rec_status = (
            "completed" if all(s == "ok" for s in b["statuses"])
            else ("partial" if any(s == "ok" for s in b["statuses"]) else "error")
        )
        auc_m, auc_s = _stat(b["aucs"])
        acc_m, acc_s = _stat(b["accs"])
        auc_tr_m, auc_tr_s = _stat(b["auc_trains"])
        acc_tr_m, acc_tr_s = _stat(b["acc_trains"])
        rmse_m, rmse_s = _stat(b["rmses"])
        mae_m, mae_s = _stat(b["maes"])
        mmd_m, mmd_s = _stat(b["mmds"])
        runtime_total = float(np.nansum([
            v for v in b["runtime_secs"] if v is not None
        ]))
        row = dict(zip(_CELL_KEY, key))
        row.update({
            "n_folds": len(b["seeds"]),
            "auc_mean": auc_m, "auc_std": auc_s,
            "acc_mean": acc_m, "acc_std": acc_s,
            "auc_train_mean": auc_tr_m, "auc_train_std": auc_tr_s,
            "acc_train_mean": acc_tr_m, "acc_train_std": acc_tr_s,
            "rmse_mean": rmse_m, "rmse_std": rmse_s,
            "mae_mean": mae_m, "mae_std": mae_s,
            "mmd_mean": mmd_m, "mmd_std": mmd_s,
            "runtime_sec_total": runtime_total,
            "status": rec_status,
        })
        pooled.append(row)
    df = pd.DataFrame(pooled)
    if not df.empty:
        df = df.sort_values(by=list(_CELL_KEY), kind="stable").reset_index(drop=True)
    return df


def _merge_into_manifest(pooled: pd.DataFrame, manifest_path: Path,
                         out_path: Path) -> int:
    """Fill ``status``, ``auc_mean``, ``acc_mean``, ``runtime_sec`` of every
    manifest row whose cell-coordinates appear in ``pooled``."""
    manifest = pd.read_csv(manifest_path)
    for col in ("status", "auc_mean", "acc_mean", "runtime_sec"):
        if col not in manifest.columns:
            manifest[col] = ""
    key_cols = list(_CELL_KEY)
    pooled_indexed = pooled.set_index(key_cols)
    n_filled = 0
    for i, row in manifest.iterrows():
        key = tuple(row[k] for k in key_cols)
        if key in pooled_indexed.index:
            p = pooled_indexed.loc[key]
            manifest.at[i, "auc_mean"] = float(p["auc_mean"]) if not np.isnan(p["auc_mean"]) else ""
            manifest.at[i, "acc_mean"] = float(p["acc_mean"]) if not np.isnan(p["acc_mean"]) else ""
            manifest.at[i, "runtime_sec"] = float(p["runtime_sec_total"])
            manifest.at[i, "status"] = str(p["status"])
            n_filled += 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out_path, index=False)
    return n_filled


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--results-dir", default="results")
    p.add_argument("--run-name", default=None,
                   help="Scope the walk to one run_name subdir; default: all.")
    p.add_argument("--out", default=None,
                   help="Where to write the per-cell summary CSV. Defaults to "
                        "<results-dir>/<run-name>/ablation_results.csv.")
    p.add_argument("--manifest", default=None,
                   help="If provided alongside --merge-into, fills metric columns "
                        "of matching manifest rows.")
    p.add_argument("--merge-into", default=None,
                   help="Output path for the manifest-merged CSV.")
    args = p.parse_args(argv)

    records = _walk_run(Path(args.results_dir), args.run_name)
    pooled = _pool_cells(records)

    out_path = Path(
        args.out
        if args.out
        else (Path(args.results_dir) / (args.run_name or "")
              / "ablation_results.csv")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if pooled.empty:
        # Emit a header-only CSV so downstream `pd.read_csv` always succeeds.
        header_cols = list(_CELL_KEY) + [
            "n_folds", "auc_mean", "auc_std", "acc_mean", "acc_std",
            "auc_train_mean", "auc_train_std", "acc_train_mean", "acc_train_std",
            "rmse_mean", "rmse_std", "mae_mean", "mae_std",
            "mmd_mean", "mmd_std", "runtime_sec_total", "status",
        ]
        pd.DataFrame(columns=header_cols).to_csv(out_path, index=False)
    else:
        pooled.to_csv(out_path, index=False)
    print(
        f"pooled {len(records):,} fold records into {len(pooled):,} cells "
        f"-> {out_path}"
    )

    if args.manifest and args.merge_into:
        n = _merge_into_manifest(pooled, Path(args.manifest), Path(args.merge_into))
        print(f"merged {n:,} cells into {args.merge_into}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
