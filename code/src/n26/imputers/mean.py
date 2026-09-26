"""MEAN: column-mean fill from training data."""
from __future__ import annotations
import warnings

import numpy as np

from n26.imputers import register_imputer
from n26.imputers.base import Imputer


@register_imputer("MEAN")
class MeanImputer(Imputer):
    """Per-column training mean. All-NaN columns fall back to 0 (with warning)."""

    def fit_transform(
        self, X_train: np.ndarray, X_test: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", r"Mean of empty slice")
            col_means = np.nanmean(X_train, axis=0)
        all_nan_cols = np.isnan(col_means)
        if all_nan_cols.any():
            warnings.warn(
                f"MEAN imputer: {int(all_nan_cols.sum())} train columns are all NaN; filling with 0."
            )
            col_means = np.where(all_nan_cols, 0.0, col_means)

        Xt = X_train.copy()
        Xe = X_test.copy()
        nan_train = np.isnan(Xt)
        nan_test = np.isnan(Xe)
        Xt = np.where(nan_train, col_means, Xt)
        Xe = np.where(nan_test, col_means, Xe)
        return Xt, Xe
