"""D3-G3 — Grade 3 ELA cohort loader.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from n26.data import register_dataset
from n26.data.base import Dataset

_ROOT = Path(__file__).resolve().parents[3]
_CSV_PATH = _ROOT.parent / "datasets" / "D3" / "D3_G3.csv"

_FEATURE_COLS = [
    "ED", "ESE", "504", "ELL",
    "Sum Days Sus", "ADA",
    "PY SS", "FAST PM1 SS", "CY IR SS PM1", "ELA QC",
]
_DEMO_COLS = ["ED", "ESE", "504", "ELL"]
_TARGET_THRESHOLD = {
    "below_level_2": ("FAST PM3 SS", 186),
    "below_level_3": ("FAST PM3 SS", 201),
}


@register_dataset("D3-G3")
def load_d3_g3() -> Dataset:
    if not _CSV_PATH.exists():
        raise FileNotFoundError(
            f"D3_G3.csv not found at {_CSV_PATH}."
        )

    df = pd.read_csv(_CSV_PATH)

    if len(df) != 1137:
        raise RuntimeError(f"D3-G3 row count {len(df)} != 1137 (spec). Did the CSV change?")

    missing_features = [c for c in _FEATURE_COLS if c not in df.columns]
    if missing_features:
        raise RuntimeError(f"D3-G3 CSV missing feature columns: {missing_features}")
    X = df[_FEATURE_COLS].astype(np.float64).to_numpy()

    y_dict = {}
    for tgt, (col, threshold) in _TARGET_THRESHOLD.items():
        if col not in df.columns:
            raise RuntimeError(f"D3-G3 CSV missing target source column: {col}")
        y_dict[tgt] = (df[col] < threshold).astype(np.int8).to_numpy()

    demo = df[_DEMO_COLS].copy()
    is_complete_case = ~np.isnan(X).any(axis=1)
    splits = {"student": KFold(n_splits=5, shuffle=True, random_state=42)}

    return Dataset(
        name="D3-G3",
        X=X,
        y_dict=y_dict,
        demo=demo,
        feature_names=_FEATURE_COLS,
        splits=splits,
        demo_axes={"primary": "ED"},
        is_complete_case=is_complete_case,
    )
