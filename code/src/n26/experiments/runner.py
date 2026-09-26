"""End-to-end runner: one cell, JSON output."""
from __future__ import annotations
from dataclasses import dataclass
import os
from pathlib import Path
import re
import time
import traceback
from typing import Any

import numpy as np

from n26.classifiers import get_classifier
from n26.data import load_dataset
from n26.experiments.schema import make_initial_record
from n26.imputers import get_imputer
from n26.metrics.classification import acc, auc
from n26.metrics.imputation import mae_on_mask, mmd_rbf, rmse_on_mask
from n26.metrics.stratified import stratified_classification, stratified_imputation
from n26.utils.standardize import Standardizer
from n26.missingness import apply_mask
from n26.utils.io import atomic_json_write
from n26.utils.logging import emit_event, progress_path_for_run
from n26.utils.seeding import derive_seed


@dataclass
class RunCellArgs:
    dataset: str
    target: str
    regime: str
    rate: int
    imputer: str
    classifier: str
    seed: int
    split: str
    out_dir: Path
    run_name: str
    experiment_id: int | None = None
    time_budget_sec: int | None = None
    overwrite: bool = False
    device: str | None = None
    train_rate: int | None = None
    complete_case_finetune: bool = False
    maskmlp_preset: str = "paper"
    include_unlabeled_in_imputer: bool = False


_MASKMLP_PRESETS: dict[str, dict] = {
    "paper": {},  
    "notebook": {
        "n_epochs": 1000,
        "do_pretrain": False,           
        "pretrain_epochs": 0,
        "lr": 5e-5,
        "weight_decay": 1e-3,
        "betas": (0.9, 0.98),
        "eps": 1e-9,
        "use_cosine_warm_restarts": True,
        "cosine_T_0": 250,
        "cosine_T_mult": 2,
        "val_size": 0.0,                 
        "patience": 0,                   
    },
}


def cell_output_path(args: RunCellArgs) -> Path:
    return (
        Path(args.out_dir)
        / args.dataset
        / args.target
        / args.regime
        / str(args.rate)
        / args.imputer
        / args.classifier
        / f"seed{args.seed}.json"
    )


def run_cell(args: RunCellArgs) -> Path:
    """Run one cell. Always writes a JSON; returns its path."""
    out_path = cell_output_path(args)
    if out_path.exists() and not args.overwrite:
        return out_path

    record = make_initial_record(
        experiment_id=args.experiment_id,
        run_name=args.run_name,
        cell={
            "dataset": args.dataset,
            "target": args.target,
            "regime": args.regime,
            "rate": args.rate,
            "imputer": args.imputer,
            "classifier": args.classifier,
            "seed": args.seed,
            "split": args.split,
            "split_fold": args.seed % 5,
            "demo_axis": None,
        },
    )
    record["timing"] = {}
    record["config"] = {
        "time_budget_sec": args.time_budget_sec,
        "device": args.device,
        "split_seed_random_state": 42,
    }
    t0 = time.perf_counter()

    def flush():
        atomic_json_write(out_path, record)

    flush()
    current_stage = "load"

    progress_log = progress_path_for_run("logs", args.run_name)
    try:
        emit_event(progress_log, "cell_start",
                   experiment_id=args.experiment_id,
                   run=args.run_name,
                   dataset=args.dataset,
                   target=args.target,
                   regime=args.regime,
                   rate=args.rate,
                   imputer=args.imputer,
                   classifier=args.classifier,
                   seed=args.seed)
    except Exception:
        pass

    try:
        ts = time.perf_counter()
        ds = load_dataset(args.dataset)
        if args.target not in ds.y_dict:
            raise ValueError(f"target '{args.target}' not in dataset '{args.dataset}' (have {ds.targets})")
        y = ds.y_dict[args.target]

        X_full = ds.X
        y_full = y
        demo_for_cell = ds.demo

        record["timing"]["load_sec"] = time.perf_counter() - ts

        splitter = ds.splits[args.split]
        folds = list(splitter.split(X_full))
        train_idx, test_idx = folds[args.seed % 5]
        X_train_orig = X_full[train_idx].copy()
        X_test_orig = X_full[test_idx].copy()
        y_train, y_test = y_full[train_idx], y_full[test_idx]
        record["data_shape"] = {
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "n_features": int(ds.d),
            "natural_missing_pct_train": float(np.isnan(X_train_orig).mean() * 100.0),
            "n_complete_cases": int(ds.is_complete_case.sum()),
        }
        flush()
        current_stage = "mask"

        ts = time.perf_counter()
        train_rate = args.train_rate if args.train_rate is not None else args.rate
        mask_seed_train = derive_seed(args.dataset, args.target, args.regime, train_rate, args.seed, "train")
        mask_seed_test = derive_seed(args.dataset, args.target, args.regime, args.rate, args.seed, "test")

        demo_train_for_mask = None
        demo_test_for_mask = None

        artificial_train = apply_mask(
            X_train_orig, regime=args.regime, rate=train_rate, seed=mask_seed_train,
            demo=demo_train_for_mask, observed_only=True,
        )
        artificial_test = apply_mask(
            X_test_orig, regime=args.regime, rate=args.rate, seed=mask_seed_test,
            demo=demo_test_for_mask, observed_only=True,
        )
        X_train_masked = X_train_orig.copy()
        X_test_masked = X_test_orig.copy()
        X_train_masked[artificial_train] = np.nan
        X_test_masked[artificial_test] = np.nan
        record["data_shape"]["artificial_missing_pct_train_observed"] = float(train_rate)
        record["data_shape"]["artificial_missing_pct_test_observed"] = float(args.rate)
        record["data_shape"]["total_missing_pct_train"] = float(np.isnan(X_train_masked).mean() * 100.0)
        record["data_shape"]["n_artificially_masked_cells_train"] = int(artificial_train.sum())
        record["data_shape"]["n_artificially_masked_cells_test"] = int(artificial_test.sum())
        record["timing"]["mask_sec"] = time.perf_counter() - ts
        flush()
        current_stage = "impute"

        scaler: Standardizer | None = None
        if args.imputer != "NATIVE":
            scaler = Standardizer().fit(X_train_masked)
            X_train_masked = scaler.transform(X_train_masked)
            X_test_masked = scaler.transform(X_test_masked)
            X_train_orig_for_metrics = scaler.transform(X_train_orig)
            X_test_orig_for_metrics = scaler.transform(X_test_orig)
        else:
            X_train_orig_for_metrics = X_train_orig
            X_test_orig_for_metrics = X_test_orig

        ts = time.perf_counter()

        # ``<base>+Raw`` suffix → force-enable include_unlab; keep the suffix in the output path.
        base_imp_name = args.imputer
        include_unlab = args.include_unlabeled_in_imputer
        if base_imp_name.endswith("+Raw"):
            base_imp_name = base_imp_name[: -len("+Raw")]
            include_unlab = True
        imp = get_imputer(base_imp_name)

        n_train_lab = X_train_masked.shape[0]
        n_unlab_added = 0
        if include_unlab and getattr(ds, "X_unlabeled", None) is not None \
                and ds.X_unlabeled.shape[0] > 0:
            X_unlab_orig = np.asarray(ds.X_unlabeled, dtype=np.float64)
            mask_seed_unlab = derive_seed(args.dataset, args.target, args.regime,
                                          train_rate, args.seed, "unlabeled")
            artificial_unlab = apply_mask(
                X_unlab_orig, regime=args.regime, rate=train_rate,
                seed=mask_seed_unlab, demo=None, observed_only=True,
            )
            X_unlab_masked = X_unlab_orig.copy()
            X_unlab_masked[artificial_unlab] = np.nan
            if scaler is not None:
                X_unlab_masked = scaler.transform(X_unlab_masked)
            X_train_for_imputer = np.vstack([X_train_masked, X_unlab_masked])
            n_unlab_added = X_unlab_masked.shape[0]
        else:
            X_train_for_imputer = X_train_masked

        record["data_shape"]["n_unlabeled_added_to_imputer"] = int(n_unlab_added)

        X_train_combined_imp, X_test_imp = imp.fit_transform(
            X_train_for_imputer, X_test_masked
        )
        # Slice back to the labeled rows so y_train alignment is preserved.
        X_train_imp = X_train_combined_imp[:n_train_lab]
        record["timing"]["impute_sec"] = time.perf_counter() - ts

        if args.imputer == "NATIVE":
            record["metrics"]["imputation"] = {
                "rmse": None, "mae": None, "mmd": None,
                "rmse_test": None, "mae_test": None, "mmd_test": None,
                "n_eval_cells_train": int(artificial_train.sum()),
                "n_eval_cells_test": int(artificial_test.sum()),
                "scale": "raw",
            }
        else:
            # MMD: restrict to rows finite in both arrays to avoid NaN propagation.
            valid_train = ~np.isnan(X_train_orig_for_metrics).any(axis=1) & ~np.isnan(X_train_imp).any(axis=1)
            mmd_train = float("nan") if valid_train.sum() < 10 else mmd_rbf(
                X_train_orig_for_metrics[valid_train], X_train_imp[valid_train]
            )
            valid_test = ~np.isnan(X_test_orig_for_metrics).any(axis=1) & ~np.isnan(X_test_imp).any(axis=1)
            mmd_test = float("nan") if valid_test.sum() < 10 else mmd_rbf(
                X_test_orig_for_metrics[valid_test], X_test_imp[valid_test]
            )
            record["metrics"]["imputation"] = {
                "rmse": rmse_on_mask(X_train_orig_for_metrics, X_train_imp, artificial_train),
                "mae": mae_on_mask(X_train_orig_for_metrics, X_train_imp, artificial_train),
                "mmd": mmd_train,
                "rmse_test": rmse_on_mask(X_test_orig_for_metrics, X_test_imp, artificial_test),
                "mae_test": mae_on_mask(X_test_orig_for_metrics, X_test_imp, artificial_test),
                "mmd_test": mmd_test,
                "n_eval_cells_train": int(artificial_train.sum()),
                "n_eval_cells_test": int(artificial_test.sum()),
                "scale": "standardized",
            }
        flush()
        current_stage = "classify"

        ts = time.perf_counter()
        clf_kwargs = {}

        # ``<classifier>:K=N`` suffix → forward n_estimators=N to the wrapper.
        clf_base_name = args.classifier
        m_clf = re.match(r"^(.+):K=(\d+)$", args.classifier)
        if m_clf:
            clf_base_name = m_clf.group(1)
            clf_kwargs["n_estimators"] = int(m_clf.group(2))

        if args.device is not None and clf_base_name == "TabPFN-v2":
            clf_kwargs["device"] = args.device
        # TabPFN-v2-FT family: per-cell checkpoints under $TABPFN_FT_CHECKPOINT_ROOT;
        # transfer variants read from the source dataset's (K, regime, seed) coordinate.
        if clf_base_name.startswith("TabPFN-v2-FT"):
            if args.device is not None:
                clf_kwargs["device"] = args.device
            ckpt_root = Path(os.environ.get(
                "TABPFN_FT_CHECKPOINT_ROOT",
                "/runpod-volume/checkpoints/tabpfn_ft",
            ))
            K_for_path = int(clf_kwargs.get("n_estimators", 1))
            cell_path = (
                f"K{K_for_path}/{args.regime}/seed{args.seed}"
            )
            if clf_base_name == "TabPFN-v2-FT":
                clf_kwargs["output_dir"] = (
                    ckpt_root / args.dataset / cell_path
                )
            elif clf_base_name == "TabPFN-v2-FT-from-D1":
                clf_kwargs["ckpt_dir"] = ckpt_root / "D1" / cell_path
            elif clf_base_name == "TabPFN-v2-FT-init-D1":
                # Two-stage FT: read D1 ckpt, write target ckpt to a separate namespace.
                clf_kwargs["ckpt_dir"] = ckpt_root / "D1" / cell_path
                clf_kwargs["output_dir"] = (
                    ckpt_root / f"{args.dataset}-init-D1" / cell_path
                )
            elif clf_base_name == "TabPFN-v2-FT-Unified":
                clf_kwargs["ckpt_dir"] = ckpt_root / "UNIFIED" / cell_path
        if clf_base_name == "MaskMLP" and args.complete_case_finetune:
            clf_kwargs["complete_case_finetune"] = True
        if clf_base_name == "MaskMLP":
            preset_kwargs = _MASKMLP_PRESETS.get(args.maskmlp_preset)
            if preset_kwargs is None:
                raise ValueError(
                    f"unknown maskmlp_preset='{args.maskmlp_preset}'; "
                    f"known: {sorted(_MASKMLP_PRESETS)}"
                )
            for k, v in preset_kwargs.items():
                clf_kwargs.setdefault(k, v)
            record["config"]["maskmlp_preset"] = args.maskmlp_preset
        clf = get_classifier(clf_base_name, **clf_kwargs)
        fit_kwargs: dict = {}
        if clf_base_name == "MaskMLP":
            # Pre-training corpus: rows outside train/test fold + ds.X_unlabeled (avoid transduction).
            in_fold_in_dsx = np.zeros(ds.X.shape[0], dtype=bool)
            in_fold_in_dsx[train_idx] = True
            in_fold_in_dsx[test_idx] = True
            extras: list[np.ndarray] = []
            extra_in_dsx = ds.X[~in_fold_in_dsx]
            if extra_in_dsx.shape[0] > 0:
                extras.append(extra_in_dsx)
            if (
                getattr(ds, "X_unlabeled", None) is not None
                and ds.X_unlabeled.shape[0] > 0
                and getattr(ds, "X_unlabeled_for_pretrain", True)
            ):
                extras.append(ds.X_unlabeled)
            if extras:
                fit_kwargs["X_pretrain"] = np.vstack(extras)
                record["data_shape"]["n_pretrain_extras"] = int(fit_kwargs["X_pretrain"].shape[0])
        clf.fit(X_train_imp, y_train, **fit_kwargs)
        record["timing"]["classify_fit_sec"] = time.perf_counter() - ts

        ts = time.perf_counter()
        proba_test = clf.predict_proba(X_test_imp)[:, 1]
        proba_train = clf.predict_proba(X_train_imp)[:, 1]
        record["timing"]["classify_predict_sec"] = time.perf_counter() - ts
        current_stage = "metrics"

        ts = time.perf_counter()
        record["metrics"]["classification"] = {
            "auc": auc(y_test, proba_test),
            "acc": acc(y_test, proba_test),
            "auc_train": auc(y_train, proba_train),
            "acc_train": acc(y_train, proba_train),
            "n_pos_test": int(y_test.sum()),
            "n_neg_test": int(len(y_test) - y_test.sum()),
        }

        primary_axis = ds.demo_axes.get("primary")
        record["subgroup_metrics"] = {}
        for axis in demo_for_cell.columns:
            demo_test = demo_for_cell[axis].iloc[test_idx].reset_index(drop=True)
            if demo_test.notna().sum() == 0:
                continue
            sg_clf = stratified_classification(y_test, proba_test, demo_test)
            if args.imputer != "NATIVE":
                sg_imp = stratified_imputation(
                    X_test_orig_for_metrics, X_test_imp, artificial_test, demo_test
                )
            else:
                sg_imp = {"groups": {}, "gap": {"rmse": None, "mae": None}}
            record["subgroup_metrics"][axis] = {
                "axis_coverage_pct_test": float(demo_test.notna().mean() * 100.0),
                "is_primary": axis == primary_axis,
                "classification": sg_clf,
                "imputation": sg_imp,
            }

        record["timing"]["metrics_sec"] = time.perf_counter() - ts

        record["status"] = "ok"
        record["timing"]["total_sec"] = time.perf_counter() - t0
        flush()
        try:
            emit_event(progress_log, "cell_done",
                       experiment_id=args.experiment_id,
                       run=args.run_name,
                       status=record.get("status", "ok"),
                       runtime_sec=record["timing"].get("total_sec", 0.0))
        except Exception:
            pass
        return out_path

    except Exception as e:
        record["status"] = "error"
        record["errors"] = {
            "kind": type(e).__name__,
            "stage": current_stage,
            "message": str(e),
            "traceback": traceback.format_exc(),
        }
        record["timing"]["total_sec"] = time.perf_counter() - t0
        flush()
        try:
            emit_event(progress_log, "cell_done",
                       experiment_id=args.experiment_id,
                       run=args.run_name,
                       status="error",
                       stage=current_stage,
                       runtime_sec=record["timing"].get("total_sec", 0.0))
        except Exception:
            pass
        raise
