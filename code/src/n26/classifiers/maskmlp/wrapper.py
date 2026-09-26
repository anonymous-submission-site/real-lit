"""MaskMLP — self-supervised MLP baseline scaffold. Recipe: Shangguan et al. (2024)."""
from __future__ import annotations

import numpy as np

from n26.classifiers import register_classifier
from n26.classifiers.base import Classifier


@register_classifier("MaskMLP")
class MaskMLPClassifier(Classifier):
    """MaskMLP wrapper. Stub: implement against the recipe documented in paper."""

    def __init__(
        self,
        device: str | None = None,
        n_estimators: int = 1,
        **_: object,
    ) -> None:
        self.device = device
        self.n_estimators = int(n_estimators)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MaskMLPClassifier":
        raise NotImplementedError(
            "MaskMLP is a scaffold in the public harness. The recipe is "
            "fully specified in the paper's Method / Appendix sections"
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise RuntimeError("MaskMLPClassifier.predict_proba called before .fit")
