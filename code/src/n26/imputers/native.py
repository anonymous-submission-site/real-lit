"""NATIVE: passthrough imputer. Classifier handles NaN itself."""
from __future__ import annotations

import numpy as np

from n26.imputers import register_imputer
from n26.imputers.base import Imputer


@register_imputer("NATIVE")
class NativeImputer(Imputer):
    """Identity: return X unchanged. Classifiers with native NaN handling
    (TabPFN-v2, TabICL-v2, TabDPT, MaskMLP) consume the result directly."""

    def fit_transform(
        self, X_train: np.ndarray, X_test: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return X_train.copy(), X_test.copy()
