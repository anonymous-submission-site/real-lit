"""Sklearn-only baseline classifiers — *not* in the paper.

These wrappers exist so the public harness produces a runnable AUC out
of the box without torch or foundation-model checkpoints. They are
**stand-ins**, not paper methods. The four paper classifiers live in
their own files (``tabpfn.py``, ``tabicl.py``, ``tabdpt.py``,
``maskmlp/wrapper.py``) and are scaffold stubs by default.

Two stand-ins are registered:

- ``HGB``       : sklearn ``HistGradientBoostingClassifier`` — handles
                  NaN inputs natively, so it composes cleanly with the
                  paper's NATIVE no-imputation baseline.
- ``LogReg``    : class-balanced L2 logistic regression — useful as a
                  trivial linear baseline for sanity checks.

To use a stand-in cell from the manifest, write the manifest with
``--classifiers HGB`` (or ``LogReg``) instead of the four paper
classifiers.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from n26.classifiers import register_classifier
from n26.classifiers.base import Classifier


@register_classifier("HGB")
class HGBClassifier(Classifier):
    """Histogram gradient-boosting trees. Stand-in only (not in paper)."""

    def __init__(
        self,
        max_iter: int = 200,
        random_state: int = 0,
        n_estimators: int | None = None,
        **_: object,
    ) -> None:
        self.max_iter = int(max_iter)
        self.random_state = int(random_state)
        self._model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HGBClassifier":
        self._model = HistGradientBoostingClassifier(
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        self._model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("HGBClassifier.predict_proba called before .fit")
        return self._model.predict_proba(X)


@register_classifier("LogReg")
class LogRegClassifier(Classifier):
    """L2 logistic regression. Stand-in only (not in paper)."""

    def __init__(
        self,
        max_iter: int = 2000,
        random_state: int = 0,
        n_estimators: int | None = None,
        **_: object,
    ) -> None:
        self.max_iter = int(max_iter)
        self.random_state = int(random_state)
        self._model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogRegClassifier":
        self._model = LogisticRegression(
            max_iter=self.max_iter,
            class_weight="balanced",
            solver="liblinear",
            random_state=self.random_state,
        )
        self._model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("LogRegClassifier.predict_proba called before .fit")
        return self._model.predict_proba(X)
