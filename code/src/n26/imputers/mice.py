"""MICE: multivariate imputation by chained equations.

Uses sklearn's IterativeImputer with BayesianRidge estimator (sklearn default),
``max_iter=10``. Fits on X_train only; transforms X_test using the fitted
state (no leakage).
"""
from __future__ import annotations

import numpy as np

from n26.imputers import register_imputer
from n26.imputers.base import Imputer


@register_imputer("MICE")
class MICEImputer(Imputer):
    """sklearn IterativeImputer wrapper. Train-only fit, no test leakage."""

    def __init__(
        self,
        max_iter: int = 10,
        random_state: int = 0,
        sample_posterior: bool = False,
    ):
        self.max_iter = max_iter
        self.random_state = random_state
        self.sample_posterior = sample_posterior

    def fit_transform(
        self, X_train: np.ndarray, X_test: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        # Required to "enable" IterativeImputer in current sklearn versions.
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer

        imp = IterativeImputer(
            max_iter=self.max_iter,
            random_state=self.random_state,
            sample_posterior=self.sample_posterior,
            keep_empty_features=True,
        )
        Xt = imp.fit_transform(X_train)
        Xe = imp.transform(X_test)
        return Xt, Xe
