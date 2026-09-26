"""Shared splitter wrappers for dataset loaders."""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupKFold


class GroupKFoldWithGroups:
    """GroupKFold with groups bound at construction, so callers can use the
    uniform ``.split(X)`` API across student- and school-based splits.
    """

    def __init__(self, splitter: GroupKFold, groups: np.ndarray):
        self._splitter = splitter
        self._groups = np.asarray(groups)

    def split(self, X, y=None):
        return self._splitter.split(X, y=y, groups=self._groups)

    def get_n_splits(self, X=None, y=None, groups=None):
        if groups is None:
            groups = self._groups
        return self._splitter.get_n_splits(X, y, groups)
