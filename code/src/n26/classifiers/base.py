"""Classifier protocol: sklearn-compatible fit/predict_proba."""
from __future__ import annotations
from abc import ABC, abstractmethod

import numpy as np


class Classifier(ABC):
    """Sklearn-style binary classifier.

    Implementations must:
      - fit on (X, y) where X may contain NaN (NATIVE handling per-classifier),
      - predict_proba returns shape (n, 2) with each row summing to 1.
    """

    name: str = "base"

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "Classifier":
        ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        ...
