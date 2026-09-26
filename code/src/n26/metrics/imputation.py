"""Imputation metrics: RMSE, MAE on artificially-masked cells."""
from __future__ import annotations

import numpy as np


def rmse_on_mask(X_true: np.ndarray, X_pred: np.ndarray, mask: np.ndarray) -> float:
    """Root-mean-squared error over masked cells; NaN if mask is empty."""
    if mask.sum() == 0:
        return float("nan")
    diff = X_true[mask] - X_pred[mask]
    return float(np.sqrt(np.mean(diff**2)))


def mae_on_mask(X_true: np.ndarray, X_pred: np.ndarray, mask: np.ndarray) -> float:
    """Mean absolute error over cells where mask is True. NaN if empty."""
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(X_true[mask] - X_pred[mask])))


def mmd_rbf(X: np.ndarray, Y: np.ndarray, bandwidth: float | None = None) -> float:
    """Squared MMD with RBF kernel; median-heuristic bandwidth when None. Biased estimator."""
    from scipy.spatial.distance import cdist

    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)

    if bandwidth is None:
        Z = np.vstack([X, Y])
        if len(Z) > 200:
            rng = np.random.default_rng(0)
            idx = rng.choice(len(Z), size=200, replace=False)
            Z = Z[idx]
        d2 = cdist(Z, Z, metric="sqeuclidean")
        d2_offdiag = d2[~np.eye(len(Z), dtype=bool)]
        median_d2 = float(np.median(d2_offdiag))
        bandwidth = float(np.sqrt(max(median_d2, 1e-12) / 2.0))

    sigma2 = bandwidth ** 2
    K_xx = np.exp(-cdist(X, X, metric="sqeuclidean") / (2.0 * sigma2))
    K_yy = np.exp(-cdist(Y, Y, metric="sqeuclidean") / (2.0 * sigma2))
    K_xy = np.exp(-cdist(X, Y, metric="sqeuclidean") / (2.0 * sigma2))

    mmd2 = K_xx.mean() + K_yy.mean() - 2.0 * K_xy.mean()
    return float(max(mmd2, 0.0))
