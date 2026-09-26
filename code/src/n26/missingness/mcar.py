"""MCAR (Missing Completely At Random) mask injection."""
from __future__ import annotations

import numpy as np


def apply_mcar(
    X: np.ndarray,
    rate: int,
    seed: int,
    observed_only: bool = True,
) -> np.ndarray:
    """Sample a uniform-Bernoulli mask at the given rate."""
    if not (0 <= rate < 100):
        raise ValueError(f"rate must be in [0, 100), got {rate}")
    if rate == 0:
        return np.zeros_like(X, dtype=bool)
    rng = np.random.default_rng(seed)
    p = rate / 100.0
    raw = rng.random(X.shape) < p
    if observed_only:
        observed = ~np.isnan(X)
        return raw & observed
    return raw
