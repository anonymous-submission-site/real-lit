"""Imputer protocol: fit_transform(X_train, X_test) -> (X_train_imp, X_test_imp)."""
from __future__ import annotations
from abc import ABC, abstractmethod

import numpy as np


class Imputer(ABC):
    """Imputer interface.

    Implementations must:
      - fit on X_train only (no test leakage),
      - transform both X_train and X_test using the fitted state,
      - return float arrays with no NaN remaining (unless the imputer
        is NATIVE/passthrough, in which case NaNs are preserved).
    """

    name: str = "base"

    @abstractmethod
    def fit_transform(
        self, X_train: np.ndarray, X_test: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        ...
