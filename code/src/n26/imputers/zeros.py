"""ZEROS: zero-fill imputer."""
from __future__ import annotations

import numpy as np

from n26.imputers import register_imputer
from n26.imputers.base import Imputer


@register_imputer("ZEROS")
class ZerosImputer(Imputer):

    def fit_transform(
        self, X_train: np.ndarray, X_test: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        Xt = X_train.copy()
        Xe = X_test.copy()
        Xt[np.isnan(Xt)] = 0.0
        Xe[np.isnan(Xe)] = 0.0
        return Xt, Xe
