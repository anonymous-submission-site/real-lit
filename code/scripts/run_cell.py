#!/usr/bin/env python3
"""Run one fold of the RealLit evaluation grid and emit a per-fold JSON record.

A "cell" is one experimental configuration:
    (dataset, target, regime, rate, imputer, classifier, K).
A "fold" is one (train, test) split within the cell, identified by an
integer ``seed`` in ``[0, n_splits * n_repeats)``. Each invocation of
``run_cell.py`` runs **one fold** and writes one JSON record at::

    <out_dir>/<run_name>/<dataset>/<target>/<regime>/<rate>/<imputer>/<classifier>/seed<seed>.json

Usage::

    # By experiment_id (looked up in the manifest):
    python scripts/run_cell.py --experiment-id 1234567890 \\
        --run-name myrun

    # By explicit coordinates (one fold = seed 0):
    python scripts/run_cell.py \\
        --dataset D2 --target LWR \\
        --regime MCAR --rate 30 \\
        --imputer MEAN --classifier HGB \\
        --K 1 --seed 0 \\
        --run-name dev_smoke
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import accuracy_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupKFold, StratifiedKFold  # noqa: E402

from n26.classifiers import get_classifier  # noqa: E402, F401
from n26.data import load_dataset  # noqa: E402
from n26.imputers import get_imputer  # noqa: E402, F401
from n26.metrics.imputation import mae_on_mask, mmd_rbf, rmse_on_mask  # noqa: E402
from n26.metrics.stratified import stratified_classification, stratified_imputation  # noqa: E402
from n26.missingness.mar import apply_mar  # noqa: E402
from n26.missingness.mcar import apply_mcar  # noqa: E402
from n26.missingness.mnar import apply_mnar  # noqa: E402

SCHEMA_VERSION = "1.0.0"
_MECH = {"MCAR": apply_mcar, "MAR": apply_mar, "MNAR": apply_mnar}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--manifest", default="docs/ablation_matrix.csv")
    p.add_argument("--experiment-id", type=int, default=None)
    p.add_argument("--dataset")
    p.add_argument("--target")
    p.add_argument("--regime", default="MCAR")
    p.add_argument("--rate", type=int, default=0)
    p.add_argument("--imputer")
    p.add_argument("--classifier")
    p.add_argument("--K", type=int, default=1,
                   help="Permutation-ensemble size (paper diagonal sweep).")
    p.add_argument("--seed", type=int, default=0,
                   help="Fold index. rep = seed // n_splits, "
                        "fold_in_rep = seed %% n_splits.")
    p.add_argument("--n-splits", type=int, default=5,
                   help="Folds per CV repetition (paper: 5).")
    p.add_argument("--run-name", default="adhoc",
                   help="Run identifier, becomes a subdir under --out-dir.")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--time-budget-sec", type=int, default=None)
    return p.parse_args(argv)


def _resolve_cell(args: argparse.Namespace) -> dict:
    if args.experiment_id is not None:
        df = pd.read_csv(args.manifest)
        match = df.loc[df["experiment_id"] == args.experiment_id]
        if match.empty:
            raise SystemExit(f"experiment_id {args.experiment_id} not in manifest")
        row = match.iloc[0]
        return {
            "experiment_id": int(row["experiment_id"]),
            "dataset": str(row["dataset"]),
            "target": str(row["target"]),
            "regime": str(row["regime"]),
            "rate": int(row["rate"]),
            "imputer": str(row["imputer"]),
            "classifier": str(row["classifier"]),
            "K": int(row["K"]),
            "seed": int(row["seed"]),
        }
    needed = ("dataset", "target", "imputer", "classifier")
    missing = [k for k in needed if getattr(args, k) is None]
    if missing:
        raise SystemExit(
            f"missing {missing}; either pass --experiment-id or all of {needed}"
        )
    return {
        "experiment_id": -1,
        "dataset": args.dataset,
        "target": args.target,
        "regime": args.regime,
        "rate": args.rate,
        "imputer": args.imputer,
        "classifier": args.classifier,
        "K": args.K,
        "seed": args.seed,
    }


def _cell_output_path(out_dir: Path, run_name: str, cell: dict) -> Path:
    return (out_dir / run_name / cell["dataset"] / cell["target"]
            / cell["regime"] / str(cell["rate"]) / cell["imputer"]
            / cell["classifier"] / f"seed{cell['seed']}.json")


def _select_one_fold(
    X: np.ndarray, y: np.ndarray, school: np.ndarray | None,
    n_splits: int, seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(train_idx, test_idx)`` for fold ``seed % n_splits`` of
    repeat ``seed // n_splits``."""
    rep = seed // n_splits
    fold_in_rep = seed % n_splits
    if school is not None:
        rng = np.random.default_rng(rep)
        order = rng.permutation(len(school))
        cv = GroupKFold(n_splits=n_splits)
        for i, (tr, te) in enumerate(cv.split(X[order], y[order], groups=school[order])):
            if i == fold_in_rep:
                return order[tr], order[te]
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=rep)
    for i, (tr, te) in enumerate(cv.split(X, y)):
        if i == fold_in_rep:
            return tr, te
    raise RuntimeError(f"could not select fold {fold_in_rep} of {n_splits}")


def _inject_missingness(
    X: np.ndarray, regime: str, rate: int, seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(X_with_extra_missing, artificial_mask)``."""
    if regime == "none" or rate == 0:
        return X, np.zeros_like(X, dtype=bool)
    if regime not in _MECH:
        raise ValueError(f"unknown regime {regime!r}; expected one of {list(_MECH)}")
    mask = _MECH[regime](X, rate=rate, seed=seed)
    X_out = X.copy()
    X_out[mask] = np.nan
    return X_out, mask


def _permutation_ensemble(
    classifier_name: str,
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_te: np.ndarray, y_te_dummy: np.ndarray,
    K: int, seed: int,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """K-permutation ensemble. Returns (proba_te, proba_tr, fit_sec, predict_sec)."""
    rng = np.random.default_rng(seed)
    n_features = X_tr.shape[1]
    proba_te = np.zeros(len(X_te), dtype=float)
    proba_tr = np.zeros(len(X_tr), dtype=float)
    fit_sec = 0.0
    predict_sec = 0.0
    for _ in range(K):
        feat_perm = rng.permutation(n_features)
        swap_labels = bool(rng.integers(0, 2))
        clf = get_classifier(classifier_name, n_estimators=1)
        y_perm = (1 - y_tr) if swap_labels else y_tr
        t0 = time.perf_counter()
        clf.fit(X_tr[:, feat_perm], y_perm)
        fit_sec += time.perf_counter() - t0
        t0 = time.perf_counter()
        p_te = np.asarray(clf.predict_proba(X_te[:, feat_perm]))[:, 1]
        p_tr = np.asarray(clf.predict_proba(X_tr[:, feat_perm]))[:, 1]
        predict_sec += time.perf_counter() - t0
        proba_te += (1.0 - p_te) if swap_labels else p_te
        proba_tr += (1.0 - p_tr) if swap_labels else p_tr
    return proba_te / K, proba_tr / K, fit_sec, predict_sec


def _provenance() -> dict:
    pkg_versions: dict[str, str] = {}
    for pkg in ("numpy", "pandas", "scikit-learn", "scipy"):
        try:
            pkg_versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            pkg_versions[pkg] = ""
    # Optional packages used by the paper-method stubs; report only if installed.
    for opt in ("torch", "tabpfn", "tabicl", "tabdpt", "miceforest"):
        try:
            pkg_versions[opt] = importlib.metadata.version(opt)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {
        "git_sha": None,
        "git_dirty": None,
        "python_version": platform.python_version(),
        "package_versions": pkg_versions,
        "host": None,
        "gpu_name": None,
        "cuda": None,
    }


def _build_subgroup_block(ds, X_true, X_imp_te, mask_test_artificial,
                           y_te, proba_te, te_idx) -> dict:
    """Per-axis classification + imputation subgroup metrics."""
    if ds.demo is None or len(ds.demo) == 0 or ds.demo.shape[1] == 0:
        return {}
    primary = (ds.demo_axes or {}).get("primary")
    test_demo = ds.demo.iloc[te_idx].reset_index(drop=True)
    out: dict[str, Any] = {}
    for axis in test_demo.columns:
        col = test_demo[axis]
        coverage = float(col.notna().mean() * 100.0)
        block: dict[str, Any] = {
            "axis_coverage_pct_test": coverage,
            "is_primary": (axis == primary),
        }
        try:
            block["classification"] = stratified_classification(y_te, proba_te, col)
        except Exception as exc:
            block["classification"] = {"error": f"{type(exc).__name__}: {exc}"}
        if mask_test_artificial is not None and mask_test_artificial.any():
            try:
                block["imputation"] = stratified_imputation(
                    X_true, X_imp_te, mask_test_artificial, col,
                )
            except Exception as exc:
                block["imputation"] = {"error": f"{type(exc).__name__}: {exc}"}
        out[axis] = block
    return out


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cell = _resolve_cell(args)

    out_path = _cell_output_path(Path(args.out_dir), args.run_name, cell)
    if out_path.exists() and not args.overwrite:
        print(f"[skip] {out_path} already exists (pass --overwrite to redo)")
        return 0
    out_path.parent.mkdir(parents=True, exist_ok=True)

    timing: dict[str, float] = {}
    errors: list[str] | None = None
    warnings: list[str] = []

    t_total = time.perf_counter()
    t_step = time.perf_counter()
    try:
        ds = load_dataset(cell["dataset"])
    except FileNotFoundError as exc:
        errors = [f"FileNotFoundError: {exc}"]
        record = _failure_record(cell, args, errors, warnings, time.perf_counter() - t_total)
        out_path.write_text(json.dumps(record, indent=2))
        print(f"[error] {cell['dataset']}: {exc}")
        return 1
    if cell["target"] not in ds.y_dict:
        raise SystemExit(
            f"target {cell['target']!r} not in dataset {cell['dataset']!r}; "
            f"available: {list(ds.y_dict)}"
        )
    X, y = ds.X, ds.y_dict[cell["target"]]
    timing["load_sec"] = time.perf_counter() - t_step

    school = None
    split_label = "student"
    if ds.demo is not None and "school" in ds.demo.columns:
        school = ds.demo["school"].to_numpy()
        split_label = "school"

    tr_idx, te_idx = _select_one_fold(X, y, school, args.n_splits, cell["seed"])
    X_tr, X_te = X[tr_idx], X[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]

    n_natural_train = int(np.isnan(X_tr).sum())
    natural_pct_train = (
        100.0 * n_natural_train / X_tr.size if X_tr.size else 0.0
    )
    n_complete_cases = int((~np.isnan(X_tr).any(axis=1)).sum())

    t_step = time.perf_counter()
    X_te_masked, mask_test_art = _inject_missingness(
        X_te, cell["regime"], cell["rate"], seed=cell["seed"] * 1_000_000 + 1,
    )
    X_tr_masked, mask_train_art = _inject_missingness(
        X_tr, cell["regime"], cell["rate"], seed=cell["seed"] * 1_000_000 + 2,
    )
    timing["mask_sec"] = time.perf_counter() - t_step

    n_train_obs = int((~np.isnan(X_tr)).sum())
    artificial_pct_train_obs = (
        100.0 * mask_train_art.sum() / n_train_obs if n_train_obs else 0.0
    )
    total_pct_train = 100.0 * np.isnan(X_tr_masked).sum() / X_tr.size

    t_step = time.perf_counter()
    imp = get_imputer(cell["imputer"])
    X_tr_imp, X_te_imp = imp.fit_transform(X_tr_masked, X_te_masked)
    timing["impute_sec"] = time.perf_counter() - t_step

    col_mean = np.nanmean(X_tr_imp, axis=0)
    col_std = np.nanstd(X_tr_imp, axis=0)
    col_std = np.where((col_std == 0.0) | np.isnan(col_std), 1.0, col_std)
    X_tr_std = (X_tr_imp - col_mean) / col_std
    X_te_std = (X_te_imp - col_mean) / col_std

    proba_te, proba_tr, fit_sec, pred_sec = _permutation_ensemble(
        cell["classifier"], X_tr_std, y_tr, X_te_std, y_te,
        K=cell["K"], seed=cell["seed"] * 1_000_001,
    )
    timing["classify_fit_sec"] = fit_sec
    timing["classify_predict_sec"] = pred_sec

    t_step = time.perf_counter()

    X_tr_true_std = (X_tr - col_mean) / col_std
    X_te_true_std = (X_te - col_mean) / col_std
    if mask_train_art.any():
        rmse_tr = rmse_on_mask(X_tr_true_std, X_tr_std, mask_train_art)
        mae_tr = mae_on_mask(X_tr_true_std, X_tr_std, mask_train_art)
    else:
        rmse_tr = float("nan"); mae_tr = float("nan")
    if mask_test_art.any():
        rmse_te = rmse_on_mask(X_te_true_std, X_te_std, mask_test_art)
        mae_te = mae_on_mask(X_te_true_std, X_te_std, mask_test_art)
    else:
        rmse_te = float("nan"); mae_te = float("nan")

    try:
        mmd_tr = mmd_rbf(X_tr_true_std, X_tr_std) if mask_train_art.any() else float("nan")
    except Exception:
        mmd_tr = float("nan")
    try:
        mmd_te = mmd_rbf(X_te_true_std, X_te_std) if mask_test_art.any() else float("nan")
    except Exception:
        mmd_te = float("nan")

    auc = float(roc_auc_score(y_te, proba_te))
    acc = float(accuracy_score(y_te, (proba_te > 0.5).astype(int)))
    auc_train = float(roc_auc_score(y_tr, proba_tr))
    acc_train = float(accuracy_score(y_tr, (proba_tr > 0.5).astype(int)))

    subgroup = _build_subgroup_block(
        ds, X_te_true_std, X_te_std, mask_test_art, y_te, proba_te, te_idx,
    )
    timing["metrics_sec"] = time.perf_counter() - t_step
    timing["total_sec"] = time.perf_counter() - t_total

    record = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": cell["experiment_id"],
        "run_name": args.run_name,
        "status": "ok",
        "cell": {
            "dataset": cell["dataset"],
            "target": cell["target"],
            "regime": cell["regime"],
            "rate": cell["rate"],
            "imputer": cell["imputer"],
            "classifier": cell["classifier"],
            "seed": cell["seed"],
            "split": split_label,
            "split_fold": cell["seed"] % args.n_splits,
            "demo_axis": (ds.demo_axes or {}).get("primary"),
        },
        "data_shape": {
            "n_train": int(len(tr_idx)),
            "n_test": int(len(te_idx)),
            "n_features": int(X.shape[1]),
            "natural_missing_pct_train": natural_pct_train,
            "n_complete_cases": n_complete_cases,
            "artificial_missing_pct_train_observed": artificial_pct_train_obs,
            "total_missing_pct_train": total_pct_train,
            "n_artificially_masked_cells_train": int(mask_train_art.sum()),
            "n_artificially_masked_cells_test": int(mask_test_art.sum()),
        },
        "metrics": {
            "imputation": {
                "rmse": rmse_tr, "mae": mae_tr, "mmd": mmd_tr,
                "rmse_test": rmse_te, "mae_test": mae_te, "mmd_test": mmd_te,
                "n_eval_cells_train": int(mask_train_art.sum()),
                "n_eval_cells_test": int(mask_test_art.sum()),
                "scale": "standardized",
            },
            "classification": {
                "auc": auc, "acc": acc,
                "auc_train": auc_train, "acc_train": acc_train,
                "n_pos_test": int(y_te.sum()),
                "n_neg_test": int(len(y_te) - y_te.sum()),
            },
        },
        "subgroup_metrics": subgroup,
        "timing": timing,
        "provenance": _provenance(),
        "config": {
            "time_budget_sec": args.time_budget_sec,
            "device": None,
            "split_seed_random_state": cell["seed"] // args.n_splits,
            "n_splits": args.n_splits,
            "K": cell["K"],
        },
        "errors": errors,
        "warnings": warnings,
    }
    out_path.write_text(json.dumps(record, indent=2, default=_json_default))

    print(
        f"[done] {cell['dataset']} / {cell['target']} / "
        f"{cell['regime']}@{cell['rate']} / {cell['imputer']} / "
        f"{cell['classifier']} / K={cell['K']} seed={cell['seed']} "
        f"-> AUC={auc:.4f} ACC={acc:.4f} ({timing['total_sec']:.1f}s) "
        f"-> {out_path}"
    )
    return 0


def _failure_record(cell, args, errors, warnings, total_sec):
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": cell["experiment_id"],
        "run_name": args.run_name,
        "status": "error",
        "cell": cell,
        "data_shape": None,
        "metrics": None,
        "subgroup_metrics": {},
        "timing": {"total_sec": total_sec},
        "provenance": _provenance(),
        "config": {
            "time_budget_sec": args.time_budget_sec,
            "device": None,
            "split_seed_random_state": cell["seed"] // args.n_splits,
            "n_splits": args.n_splits,
            "K": cell["K"],
        },
        "errors": errors,
        "warnings": warnings,
    }


def _json_default(o):
    """JSON serializer fallback for numpy scalars and NaN floats."""
    if isinstance(o, float) and np.isnan(o):
        return float("nan")
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(f"not JSON-serializable: {type(o).__name__}")


if __name__ == "__main__":
    raise SystemExit(main())
