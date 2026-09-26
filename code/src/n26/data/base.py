"""Dataset dataclass — the unit of data passed between subsystems."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass
class Dataset:
    """Loaded dataset. NumPy arrays passed in are NOT copied; mutators must .copy()."""

    name: str
    X: np.ndarray
    y_dict: Mapping[str, np.ndarray]
    demo: pd.DataFrame
    feature_names: Sequence[str]
    splits: Mapping[str, object]
    demo_axes: Mapping[str, str]
    is_complete_case: np.ndarray
    X_unlabeled: np.ndarray | None = None
    X_unlabeled_for_pretrain: bool = True

    def __post_init__(self):
        n = self.X.shape[0]
        for tgt, y in self.y_dict.items():
            if len(y) != n:
                raise ValueError(
                    f"y_dict['{tgt}'] has length {len(y)} but X has {n} rows"
                )
            unique = (
                np.unique(np.asarray(y)[~np.isnan(np.asarray(y, dtype=float))])
                if np.issubdtype(np.asarray(y).dtype, np.floating)
                else np.unique(y)
            )
            if not set(unique.tolist()).issubset({0, 1}):
                raise ValueError(
                    f"y_dict['{tgt}'] has non-binary values {unique.tolist()}; "
                    f"binary labels (0/1) required"
                )
        if len(self.demo) != n:
            raise ValueError(
                f"demo has {len(self.demo)} rows but X has {n}"
            )
        if self.is_complete_case.shape != (n,):
            raise ValueError(
                f"is_complete_case has shape {self.is_complete_case.shape}, expected ({n},)"
            )
        if self.is_complete_case.dtype != np.bool_:
            raise ValueError(
                f"is_complete_case dtype is {self.is_complete_case.dtype}, expected bool"
            )
        if len(self.feature_names) != self.X.shape[1]:
            raise ValueError(
                f"feature_names has {len(self.feature_names)} entries but X has {self.X.shape[1]} cols"
            )
        if self.X_unlabeled is not None and self.X_unlabeled.shape[1] != self.X.shape[1]:
            raise ValueError(
                f"X_unlabeled has {self.X_unlabeled.shape[1]} cols but X has "
                f"{self.X.shape[1]}; column order/feature set must match"
            )

    @property
    def targets(self) -> list[str]:
        return list(self.y_dict.keys())

    @property
    def n(self) -> int:
        return self.X.shape[0]

    @property
    def d(self) -> int:
        return self.X.shape[1]
