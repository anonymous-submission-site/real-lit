#!/usr/bin/env python3
"""Smallest end-to-end demo of the RealLit harness.

Runs one 5-fold pass for cell:
    dataset    = D2
    target     = LWR (KTEA-3 Letter and Word Recognition < 90 at end of K)
    regime     = native missingness (no extra injection)
    rate       = 0
    imputer    = MEAN
    classifier = HGB (sklearn HistGradientBoosting, for this demo)
    K          = 1

5 invocations of run_cell.py (one per fold), then aggregate.py pools
the per-fold JSONs into one ``ablation_results.csv`` row showing the
mean +/- std AUC.

The four paper classifiers (MaskMLP, TabPFN-v2, TabICL-v2, TabDPT) are
scaffold stubs that raise NotImplementedError when invoked. HGB is a
sklearn-only stand-in registered so this demo runs without external
dependencies.

Usage::

    python scripts/examples/run_minimal.py
"""
from __future__ import annotations

import sys
from pathlib import Path

HARNESS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HARNESS_ROOT / "scripts"))

from aggregate import main as aggregate_main  # noqa: E402
from run_cell import main as run_cell_main  # noqa: E402

RUN_NAME = "minimal_demo"
RESULTS_DIR = HARNESS_ROOT / "results"


def main() -> int:
    base_argv = [
        "--dataset", "D2",
        "--target", "LWR",
        "--regime", "none",
        "--rate", "0",
        "--imputer", "MEAN",
        "--classifier", "HGB",
        "--K", "1",
        "--n-splits", "5",
        "--run-name", RUN_NAME,
        "--out-dir", str(RESULTS_DIR),
        "--overwrite",
    ]
    print(f"-- running 5 folds (seed 0..4) under run_name={RUN_NAME} --")
    for seed in range(5):
        rc = run_cell_main(base_argv + ["--seed", str(seed)])
        if rc != 0:
            return rc
    print()
    print("-- aggregating per-fold JSONs into per-cell summary --")
    return aggregate_main([
        "--results-dir", str(RESULTS_DIR),
        "--run-name", RUN_NAME,
    ])


if __name__ == "__main__":
    raise SystemExit(main())
