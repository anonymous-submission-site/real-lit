"""CFMI — conditional flow-matching imputer scaffold. Reference: Simkus et al. (2025)."""
from __future__ import annotations

import numpy as np

from n26.imputers import register_imputer
from n26.imputers.base import Imputer


@register_imputer("CFMI")
class CFMIImputer(Imputer):
    """CFMI wrapper. Requires the public CFMI reference implementation."""

    def __init__(self, **_: object) -> None:
        pass

    def fit_transform(
        self, X_train: np.ndarray, X_test: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError(
            "CFMI is a scaffold in the public harness (one of the K-sweep "
            "learned imputers in the paper's Method section). Obtain the public "
            "reference implementation from Simkus et al. 2025 and "
            "replace this stub's .fit_transform with the conditional "
            "flow-matching imputer fit on the training fold's observed "
            "entries only."
        )
