"""MIRI — rectified-flow tabular imputer scaffold. Reference: Yu et al. (2025)."""
from __future__ import annotations

import numpy as np

from n26.imputers import register_imputer
from n26.imputers.base import Imputer


@register_imputer("MIRI")
class MIRIImputer(Imputer):
    """MIRI wrapper. Requires the public MIRI reference implementation."""

    def __init__(self, **_: object) -> None:
        pass

    def fit_transform(
        self, X_train: np.ndarray, X_test: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError(
            "MIRI is a scaffold in the public harness (one of the K-sweep "
            "learned imputers in the paper's Method section). Obtain the public "
            "reference implementation from Yu et al. 2025 and replace "
            "this stub's .fit_transform with the rectified-flow imputer "
            "fit on the training fold's observed entries only."
        )
