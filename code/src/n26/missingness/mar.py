"""MAR mask via per-column logistic-of-other-feature (Schouten 2018, Muzellec et al. 2020)."""
from __future__ import annotations

import numpy as np

from n26.missingness._common import _logistic_bernoulli_mask


def apply_mar(
    X: np.ndarray,
    rate: int,
    seed: int,
    observed_only: bool = True,
) -> np.ndarray:
    """Build a MAR mask. See module docstring for the protocol."""
    n, d = X.shape
    if d < 2:
        raise ValueError(f"MAR requires d>=2 columns, got d={d}")
    rng = np.random.default_rng(seed)

    x_std = X.astype(np.float64)
    col_mean = np.nanmean(x_std, axis=0)
    col_std = np.nanstd(x_std, axis=0)
    col_std = np.where((col_std == 0.0) | np.isnan(col_std), 1.0, col_std)
    x_std = (x_std - col_mean) / col_std
    x_std = np.where(np.isnan(x_std), 0.0, x_std)

    logits = np.zeros((n, d), dtype=np.float64)
    cond_idx = np.empty(d, dtype=np.int64)
    for j in range(d):
        choices = np.array([k for k in range(d) if k != j])
        cond_idx[j] = int(rng.choice(choices))
        w = float(rng.standard_normal())
        logits[:, j] = w * x_std[:, cond_idx[j]]

    observed = ~np.isnan(X) if observed_only else np.ones_like(X, dtype=bool)

    return _logistic_bernoulli_mask(logits, observed, rate=rate, seed=seed + 1)
