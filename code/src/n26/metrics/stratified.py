"""Stratified metrics: per-subgroup AUC/ACC/RMSE/MAE with n<5 suppression.

Cells where the smaller class has zero examples still yield None (AUC
undefined for single-class groups).
"""
from __future__ import annotations
from typing import Any

import numpy as np
import pandas as pd

from n26.metrics.classification import acc, auc
from n26.metrics.imputation import mae_on_mask, rmse_on_mask

_MIN_N = 5


def _safe_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float | None:
    if len(y_true) < _MIN_N:
        return None
    classes, counts = np.unique(y_true, return_counts=True)
    if len(classes) < 2 or counts.min() < _MIN_N:
        return None
    return auc(y_true, y_proba)


def _safe_acc(y_true: np.ndarray, y_proba: np.ndarray) -> float | None:
    if len(y_true) < _MIN_N:
        return None
    return acc(y_true, y_proba)


def _gap(values: list[float | None]) -> float | None:
    populated = [v for v in values if v is not None]
    if len(populated) < 2:
        return None
    return float(max(populated) - min(populated))


def stratified_classification(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    demo: pd.Series,
) -> dict[str, Any]:
    """Compute AUC/ACC stratified by levels of `demo`. Returns dict with
    `groups` (per-level dict) and `gap` (max−min across populated levels)."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    demo = pd.Series(demo).reset_index(drop=True)
    if len(demo) != len(y_true):
        raise ValueError(f"demo len {len(demo)} != y_true len {len(y_true)}")

    groups: dict[str, dict[str, Any]] = {}
    aucs: list[float | None] = []
    accs: list[float | None] = []
    for level, idx in demo.dropna().groupby(demo).groups.items():
        idx_arr = np.asarray(list(idx))
        y_g = y_true[idx_arr]
        p_g = y_proba[idx_arr]
        a = _safe_auc(y_g, p_g)
        c = _safe_acc(y_g, p_g)
        groups[str(level)] = {
            "n": int(len(idx_arr)),
            "n_pos": int(y_g.sum()),
            "auc": a,
            "acc": c,
        }
        aucs.append(a)
        accs.append(c)
    return {"groups": groups, "gap": {"auc": _gap(aucs), "acc": _gap(accs)}}


def stratified_imputation(
    X_true: np.ndarray,
    X_pred: np.ndarray,
    mask: np.ndarray,
    demo: pd.Series,
) -> dict[str, Any]:
    """Compute RMSE/MAE stratified by levels of `demo`. The mask is restricted
    to the subgroup's rows when computing per-group scores. Cells with no
    mask in a subgroup → rmse/mae null."""
    demo = pd.Series(demo).reset_index(drop=True)
    if len(demo) != X_true.shape[0]:
        raise ValueError(f"demo len {len(demo)} != X_true rows {X_true.shape[0]}")

    groups: dict[str, dict[str, Any]] = {}
    rmses: list[float | None] = []
    maes: list[float | None] = []
    for level, idx in demo.dropna().groupby(demo).groups.items():
        idx_arr = np.asarray(list(idx))
        Xt = X_true[idx_arr]
        Xp = X_pred[idx_arr]
        m = mask[idx_arr]
        if m.sum() < 1:
            r, mae = None, None
        else:
            r_val = rmse_on_mask(Xt, Xp, m)
            mae_val = mae_on_mask(Xt, Xp, m)
            r = float(r_val) if not np.isnan(r_val) else None
            mae = float(mae_val) if not np.isnan(mae_val) else None
        groups[str(level)] = {
            "n": int(len(idx_arr)),
            "n_eval_cells": int(m.sum()),
            "rmse": r,
            "mae": mae,
        }
        rmses.append(r)
        maes.append(mae)
    return {"groups": groups, "gap": {"rmse": _gap(rmses), "mae": _gap(maes)}}
