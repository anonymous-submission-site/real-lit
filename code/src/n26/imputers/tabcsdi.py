"""TabCSDI — conditional score-based diffusion imputer scaffold. Reference: Zheng et al. (2022)."""
from __future__ import annotations

import numpy as np

from n26.imputers import register_imputer
from n26.imputers.base import Imputer


@register_imputer("TabCSDI")
class TabCSDIImputer(Imputer):
    """TabCSDI wrapper. Requires the public TabCSDI reference implementation."""

    def __init__(self, **_: object) -> None:
        pass

    def fit_transform(
        self, X_train: np.ndarray, X_test: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError(
            "TabCSDI is a scaffold in the public harness (one of the K-sweep "
            "learned imputers in the paper's Method section). Obtain the public "
            "reference implementation from Zheng et al. 2022 and replace "
            "this stub's .fit_transform with the conditional score-based "
            "diffusion imputer fit on the training fold's observed "
            "entries only."
        )
