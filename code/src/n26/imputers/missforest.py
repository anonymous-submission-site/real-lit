"""MissForest: random-forest imputation via the miceforest library.

5 iterations, 100 trees per iteration. Train-only fit; test transformed via
the fitted kernel (no leakage). miceforest is imported lazily inside
fit_transform so the wrapper imports without the dep.
"""
from __future__ import annotations

import numpy as np

from n26.imputers import register_imputer
from n26.imputers.base import Imputer


@register_imputer("MissForest")
class MissForestImputer(Imputer):
    """miceforest random-forest imputer. Lazy import."""

    def __init__(self, iterations: int = 5, num_trees: int = 100, random_state: int = 0):
        self.iterations = iterations
        self.num_trees = num_trees
        self.random_state = random_state

    def fit_transform(
        self, X_train: np.ndarray, X_test: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        import miceforest as mf
        import pandas as pd

        # miceforest v6 requires string column names
        n_cols = X_train.shape[1]
        col_names = [f"c{i}" for i in range(n_cols)]

        # Pre-fill sparse columns so miceforest has enough observed values per column.
        MIN_OBSERVED_FOR_MICE = 5
        X_train = X_train.copy()
        X_test = X_test.copy()
        obs_count = (~np.isnan(X_train)).sum(axis=0)
        sparse_cols = np.where(obs_count < MIN_OBSERVED_FOR_MICE)[0]
        for c in sparse_cols:
            observed = X_train[~np.isnan(X_train[:, c]), c]
            fill = float(observed.mean()) if observed.size > 0 else 0.0
            X_train[np.isnan(X_train[:, c]), c] = fill
            X_test[np.isnan(X_test[:, c]), c] = fill

        df_train = pd.DataFrame(X_train, columns=col_names)
        # mean_match_candidates=0 → use model predictions directly (no donor matching).
        kernel = mf.ImputationKernel(
            df_train,
            num_datasets=1,
            random_state=self.random_state,
            mean_match_candidates=0,
        )
        kernel.mice(iterations=self.iterations, num_trees=self.num_trees)
        Xt = kernel.complete_data(dataset=0).to_numpy()

        df_test = pd.DataFrame(X_test, columns=col_names)
        Xe = kernel.impute_new_data(df_test).complete_data(dataset=0).to_numpy()

        return Xt.astype(np.float64), Xe.astype(np.float64)
