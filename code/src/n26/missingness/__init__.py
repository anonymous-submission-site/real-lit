"""Missingness regime dispatcher: MCAR, MAR, MNAR."""
from __future__ import annotations

import numpy as np
import pandas as pd

from n26.missingness.mar import apply_mar
from n26.missingness.mcar import apply_mcar
from n26.missingness.mnar import apply_mnar

_VALID_REGIMES = {"MCAR", "MAR", "MNAR"}


def apply_mask(
    X: np.ndarray,
    regime: str,
    rate: int,
    seed: int,
    demo: pd.Series | None = None,
    observed_only: bool = True,
) -> np.ndarray:
    """Dispatch to apply_mcar / apply_mar / apply_mnar by regime name."""
    if regime not in _VALID_REGIMES:
        raise ValueError(
            f"unknown regime '{regime}'; valid: {sorted(_VALID_REGIMES)}"
        )
    if regime == "MCAR":
        return apply_mcar(X, rate=rate, seed=seed, observed_only=observed_only)
    if regime == "MAR":
        return apply_mar(X, rate=rate, seed=seed, observed_only=observed_only)
    if regime == "MNAR":
        return apply_mnar(X, rate=rate, seed=seed, observed_only=observed_only)
    raise AssertionError(f"unreachable: {regime}")  # already validated above


__all__ = ["apply_mask"]
