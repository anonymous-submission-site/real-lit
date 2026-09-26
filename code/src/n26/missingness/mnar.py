"""MNAR mask via per-column logistic-of-own-value (Schouten 2018, Muzellec et al. 2020)."""
from __future__ import annotations

import numpy as np

from n26.missingness._common import _logistic_bernoulli_mask


def apply_mnar(
    X: np.ndarray,
    rate: int,
    seed: int,
    observed_only: bool = True,
) -> np.ndarray:
    """Build an MNAR mask. See module docstring for the protocol."""
    n, d = X.shape
    rng = np.random.default_rng(seed)

    x_std = X.astype(np.float64)
    col_mean = np.nanmean(x_std, axis=0)
    col_std = np.nanstd(x_std, axis=0)
    col_std[col_std == 0.0] = 1.0
    x_std = (x_std - col_mean) / col_std
    x_std = np.where(np.isnan(x_std), 0.0, x_std)

    # Sign of w randomises whether high or low values are preferentially missing.
    w = rng.standard_normal(d)
    logits = x_std * w

    observed = ~np.isnan(X) if observed_only else np.ones_like(X, dtype=bool)

    return _logistic_bernoulli_mask(logits, observed, rate=rate, seed=seed + 1)
