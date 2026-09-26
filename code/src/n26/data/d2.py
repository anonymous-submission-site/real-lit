"""D2 — Kindergarten Fall + End-of-Year cohort loader.

Fall-of-kindergarten predictors; binary at-risk targets at the
end-of-year KTEA-3 SS < 90 cut. Primary subgroup axis: language.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from n26.data import register_dataset
from n26.data.base import Dataset

_ROOT = Path(__file__).resolve().parents[3]
_CSV_PATH = _ROOT.parent / "datasets" / "D2" / "D2.csv"
_COHORT = 2

_FEATURE_COLS = [
    "rvo_kg_fall", "evo_kg_fall", "lcp_kg_fall",
    "ble_kg_fall", "del_kg_fall", "syn_kg_fall",
    "nwr_kg_fall", "srt_kg_fall",
    "lnc_kg_fall", "lsc_kg_fall", "rao_kg_fall",
    "wppsi4_rv_ss", "wppsi4_pn_ss", "wppsi4_mr_ss",
]
_DEMO_COLS = ["gender", "ethnicity", "race", "language"]
_TARGET_COLS = {
    "NWD": ("ktea3_nwd_ss", 90),
    "LWR": ("ktea3_lwr_ss", 90),
}


@register_dataset("D2")
def load_d2() -> Dataset:
    if not _CSV_PATH.exists():
        raise FileNotFoundError(f"D2.csv not found at {_CSV_PATH}.")

    df_raw = pd.read_csv(_CSV_PATH)
    # shipped per-cohort file: already only this cohort, no `cohort` column
    df_raw = df_raw.reset_index(drop=True)

    valid_mask = np.ones(len(df_raw), dtype=bool)
    for col, _ in _TARGET_COLS.values():
        if col not in df_raw.columns:
            raise RuntimeError(f"D2 CSV missing target column: {col}")
        valid_mask &= df_raw[col].notna().to_numpy()
    df = df_raw.loc[valid_mask].reset_index(drop=True)
    n = len(df)
    if n < 100:
        raise RuntimeError(
            f"D2 labeled cohort has only {n} rows; "
            f"check that D2.csv is the released cohort file."
        )

    missing_features = [c for c in _FEATURE_COLS if c not in df.columns]
    if missing_features:
        raise RuntimeError(f"D2 CSV missing feature columns: {missing_features}")
    X = df[_FEATURE_COLS].astype(np.float64).to_numpy()

    y_dict = {}
    for tgt, (col, threshold) in _TARGET_COLS.items():
        y_dict[tgt] = (df[col] < threshold).astype(np.int8).to_numpy()

    demo_present = [c for c in _DEMO_COLS if c in df.columns]
    demo = df[demo_present].copy()

    is_complete_case = ~np.isnan(X).any(axis=1)

    splits = {"student": KFold(n_splits=5, shuffle=True, random_state=42)}

    return Dataset(
        name="D2",
        X=X,
        y_dict=y_dict,
        demo=demo,
        feature_names=_FEATURE_COLS,
        splits=splits,
        demo_axes={"primary": "language"},
        is_complete_case=is_complete_case,
    )
