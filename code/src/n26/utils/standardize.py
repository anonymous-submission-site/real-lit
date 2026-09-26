"""Per-column NaN-aware z-score standardizer (zero-variance std clamped to 1)."""
from __future__ import annotations
import warnings

import numpy as np


class Standardizer:
    def __init__(self):
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "Standardizer":
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", r"Mean of empty slice")
            warnings.filterwarnings("ignore", r"Degrees of freedom <= 0 for slice")
            self._mean = np.nanmean(X, axis=0)
            self._std = np.nanstd(X, axis=0, ddof=0)
        self._mean = np.where(np.isnan(self._mean), 0.0, self._mean)
        self._std = np.where(
            np.isnan(self._std) | (self._std == 0.0), 1.0, self._std
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._mean is None:
            raise RuntimeError("Standardizer not fitted; call fit() first.")
        return (X - self._mean) / self._std

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        if self._mean is None:
            raise RuntimeError("Standardizer not fitted; call fit() first.")
        return X * self._std + self._mean
