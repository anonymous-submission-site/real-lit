"""Classification metrics: AUC, ACC."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score


def auc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """ROC AUC. Returns NaN when only one class is present in y_true."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_proba))


def acc(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> float:
    """Accuracy at the given probability threshold (default 0.5).
    `proba > threshold` predicts class 1; `proba <= threshold` predicts 0.
    """
    y_pred = (y_proba > threshold).astype(int)
    return float(accuracy_score(y_true, y_pred))
