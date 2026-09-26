"""TabICL-v2 lazy-import wrapper. Reference: Qu et al. (2025)."""
from __future__ import annotations

import numpy as np

from n26.classifiers import register_classifier
from n26.classifiers.base import Classifier


@register_classifier("TabICL-v2")
class TabICLClassifier(Classifier):
    """TabICL-v2 wrapper. Requires the ``tabicl`` package."""

    def __init__(
        self,
        device: str | None = None,
        n_estimators: int = 1,
        **_: object,
    ) -> None:
        self.device = device
        self.n_estimators = int(n_estimators)
        self._model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TabICLClassifier":
        try:
            from tabicl import TabICLClassifier as _TabICL
        except ImportError as exc:
            raise NotImplementedError(
                "TabICL-v2 requires the public `tabicl` package "
                "(Qu et al. 2025)."
            ) from exc
        self._model = _TabICL(device=self.device, n_estimators=self.n_estimators)
        self._model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("TabICLClassifier.predict_proba called before .fit")
        return np.asarray(self._model.predict_proba(X))
