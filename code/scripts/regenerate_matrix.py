#!/usr/bin/env python3
"""Regenerate docs/ablation_matrix.csv from the paper's evaluation grid.

The CSV is the source of truth for what cells the harness should run.
One row per (dataset, target, regime, rate, imputer, classifier, K, seed)
cell. Each row carries a stable ``experiment_id`` that is a deterministic
hash of those coordinates; re-running the script with the same arguments
produces a byte-identical CSV.

Default grid mirrors the paper's Experimental Setup:

    classifiers   = MaskMLP, TabDPT, TabICL-v2, TabPFN-v2          (4)
    K-sweep imps  = MIRI, TabCSDI, DiffPuter, CFMI                 (4)
    K values      = 1, 2, 4, 8, 16, 32                              (6)
    mechanisms    = MCAR, MAR, MNAR                                 (3)
    injected rate = 10, 20, 30                                      (3)
    references    = MICE, MissForest, NATIVE, MEAN, ZEROS           (5)
                    -> at native missingness (rate=0, K=1) only

The ``seed`` column is the **fold index**. ``run_cell.py`` runs one
fold per invocation; each cell therefore expands into ``--seeds`` rows
in the manifest. Use ``--seeds 5`` for a single 5-fold pass (fast
dev), ``--seeds 50`` for the paper's full 10x5 protocol.

Each row is annotated with ``requires_external_data`` (True for D1 and
D3 cohorts, since those are not redistributed by RealLit and must be
obtained separately).

Usage::

    python scripts/regenerate_matrix.py \\
        --datasets D1,D2,D3-G1-2,D3-G3,D4 \\
        --seeds 5 \\
        --out docs/ablation_matrix.csv
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from n26.data import list_datasets, load_dataset  # noqa: E402

DEFAULT_DATASETS = ["D1", "D2", "D3-G1-2", "D3-G3", "D4"]
DEFAULT_CLASSIFIERS = ["MaskMLP", "TabDPT", "TabICL-v2", "TabPFN-v2"]
DEFAULT_KSWEEP_IMPUTERS = ["MIRI", "TabCSDI", "DiffPuter", "CFMI"]
DEFAULT_REFERENCE_IMPUTERS = ["MICE", "MissForest", "NATIVE", "MEAN", "ZEROS"]
DEFAULT_REGIMES = ["MCAR", "MAR", "MNAR"]
DEFAULT_RATES = [10, 20, 30]
DEFAULT_K_VALUES = [1, 2, 4, 8, 16, 32]
DEFAULT_SEEDS = 5

EXTERNAL_DATASETS = {"D1", "D3-G1-2", "D3-G3", "D3-G1-2-CROSS"}


def _stable_experiment_id(coords: tuple) -> int:
    """Hash a cell tuple to a stable 32-bit unsigned int."""
    h = hashlib.sha256("|".join(str(x) for x in coords).encode()).digest()
    return int.from_bytes(h[:4], "big")


def _csv_str_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _csv_int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def _dataset_targets(name: str) -> list[str]:
    """Return the list of targets for a registered dataset.

    For external datasets we cannot call load_dataset (data may be
    missing), so we fall back to a hard-coded mapping that mirrors the
    loaders' ``_TARGET_COLS`` definitions.
    """
    if name not in EXTERNAL_DATASETS:
        return list(load_dataset(name).targets)
    return {
        "D1":               ["WordID", "WordAtk"],
        "D3-G1-2":          ["below_level_3_fast", "below_grade_iready"],
        "D3-G1-2-CROSS":    ["crossed_proficiency"],
        "D3-G3":            ["below_level_2", "below_level_3"],
    }[name]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--datasets",     default=",".join(DEFAULT_DATASETS))
    p.add_argument("--classifiers",  default=",".join(DEFAULT_CLASSIFIERS))
    p.add_argument("--imputers",     default=",".join(DEFAULT_KSWEEP_IMPUTERS),
                   help="K-sweep imputers (run at all K and all rates).")
    p.add_argument("--reference-imputers", default=",".join(DEFAULT_REFERENCE_IMPUTERS),
                   help="Reference imputers (run only at rate=0, K=1).")
    p.add_argument("--regimes",      default=",".join(DEFAULT_REGIMES))
    p.add_argument("--rates",        default=",".join(str(r) for r in DEFAULT_RATES))
    p.add_argument("--k-values",     default=",".join(str(k) for k in DEFAULT_K_VALUES))
    p.add_argument("--seeds", type=int, default=DEFAULT_SEEDS,
                   help="Number of fold rows per cell (seed in 0..seeds-1; "
                        "each row = one CV fold for run_cell.py to evaluate).")
    p.add_argument("--out",          default="docs/ablation_matrix.csv")
    args = p.parse_args(argv)

    datasets = _csv_str_list(args.datasets)
    classifiers = _csv_str_list(args.classifiers)
    ksweep_imps = _csv_str_list(args.imputers)
    ref_imps = _csv_str_list(args.reference_imputers)
    regimes = _csv_str_list(args.regimes)
    rates = _csv_int_list(args.rates)
    k_values = _csv_int_list(args.k_values)
    seeds = list(range(args.seeds))

    rows = []
    for ds_name in datasets:
        if ds_name not in list_datasets():
            raise SystemExit(
                f"unknown dataset {ds_name!r}; known: {list_datasets()}"
            )
        targets = _dataset_targets(ds_name)
        external = ds_name in EXTERNAL_DATASETS
        for target in targets:
            for imputer in ksweep_imps:
                for classifier in classifiers:
                    for regime in regimes:
                        for rate in rates:
                            for K in k_values:
                                for seed in seeds:
                                    coords = (ds_name, target, regime, rate,
                                              imputer, classifier, K, seed)
                                    rows.append({
                                        "experiment_id": _stable_experiment_id(coords),
                                        "dataset": ds_name,
                                        "target": target,
                                        "regime": regime,
                                        "rate": rate,
                                        "imputer": imputer,
                                        "classifier": classifier,
                                        "K": K,
                                        "seed": seed,
                                        "requires_external_data": external,
                                        "status": "pending",
                                        "auc_mean": "",
                                        "acc_mean": "",
                                        "runtime_sec": "",
                                        "notes": "",
                                    })
            for imputer in ref_imps:
                for classifier in classifiers:
                    for seed in seeds:
                        coords = (ds_name, target, "none", 0,
                                  imputer, classifier, 1, seed)
                        rows.append({
                            "experiment_id": _stable_experiment_id(coords),
                            "dataset": ds_name,
                            "target": target,
                            "regime": "none",
                            "rate": 0,
                            "imputer": imputer,
                            "classifier": classifier,
                            "K": 1,
                            "seed": seed,
                            "requires_external_data": external,
                            "status": "pending",
                            "auc_mean": "",
                            "acc_mean": "",
                            "runtime_sec": "",
                            "notes": "",
                        })

    df = pd.DataFrame(rows)

    df = df.sort_values(
        by=["dataset", "target", "regime", "rate", "imputer",
            "classifier", "K", "seed"],
        kind="stable",
    ).reset_index(drop=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"wrote {len(df):,} cells to {out_path}")
    n_external = int(df["requires_external_data"].sum())
    n_releasable = len(df) - n_external
    print(f"  releasable (D2/D4):     {n_releasable:,} cells")
    print(f"  requires external data: {n_external:,} cells "
          f"({sorted(EXTERNAL_DATASETS & set(df['dataset']))})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
