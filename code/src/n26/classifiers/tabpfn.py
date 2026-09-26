"""TabPFN-v2 lazy-import wrapper. Reference: Hollmann et al. (2025)."""
from __future__ import annotations

import numpy as np

from n26.classifiers import register_classifier
from n26.classifiers.base import Classifier


@register_classifier("TabPFN-v2")
class TabPFNClassifier(Classifier):
    """TabPFN-v2 wrapper. Requires the ``tabpfn`` package."""

    def __init__(
        self,
        device: str | None = None,
        n_estimators: int = 4,
        **_: object,
    ) -> None:
        self.device = device
        self.n_estimators = int(n_estimators)
        self._model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TabPFNClassifier":
        try:
            from tabpfn import TabPFNClassifier as _TabPFN
        except ImportError as exc:
            raise NotImplementedError(
                "TabPFN-v2 requires the public `tabpfn` package "
                "(Hollmann et al. 2025)"
            ) from exc
        self._model = _TabPFN(
            device=self.device, n_estimators=self.n_estimators,
        )
        self._model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("TabPFNClassifier.predict_proba called before .fit")
        return np.asarray(self._model.predict_proba(X))
