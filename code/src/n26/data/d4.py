"""D4 loader. Early-elementary feature window; binary at-risk targets
from G3 standard scores at the < 90 cut. Race and sex exposed as
demographic columns.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from n26.data import register_dataset
from n26.data.base import Dataset

_ROOT = Path(__file__).resolve().parents[3]
_CSV_PATH = _ROOT.parent / "datasets" / "D4" / "D4.csv"
_COHORT = 4

_TARGET_THRESHOLD = {
    "WID": ("G3_WJAIII_WidSS", 90),
    "WAtk": ("G3_WJAIII_WAttackSS", 90),
    "Spell": ("G3_WRAT3Spell_SS", 90),
}

_DEMO_CANDIDATES = {
    "G1_Race": "race",
    "G1_Sex": "sex",
    "Race": "race",
    "Sex": "sex",
}


@register_dataset("D4")
def load_d4() -> Dataset:
    if not _CSV_PATH.exists():
        raise FileNotFoundError(f"D4.csv not found at {_CSV_PATH}.")

    df_raw = pd.read_csv(_CSV_PATH)
    # shipped per-cohort file: already only this cohort, no `cohort` column
    df_raw = df_raw.reset_index(drop=True)

    target_cols = [col for col, _ in _TARGET_THRESHOLD.values()]
    missing_targets = [c for c in target_cols if c not in df_raw.columns]
    if missing_targets:
        raise RuntimeError(f"D4 CSV missing target columns: {missing_targets}")
    label_mask = df_raw[target_cols].notna().all(axis=1)
    df = df_raw.loc[label_mask].reset_index(drop=True)
    if len(df) < 100:
        raise RuntimeError(
            f"D4 G3-valid cohort has only {len(df)} rows; "
            f"check that D4.csv is the released cohort file."
        )

    feature_cols = sorted([c for c in df.columns if c.startswith("G1_") or c.startswith("G1 ")])
    if len(feature_cols) < 10:
        raise RuntimeError(
            f"D4 has only {len(feature_cols)} early-grade columns; "
            f"check column-naming convention with `head -1 <csv>`."
        )
    X = df[feature_cols].astype(np.float64).to_numpy()

    y_dict: dict[str, np.ndarray] = {}
    for tgt, (col, threshold) in _TARGET_THRESHOLD.items():
        y_dict[tgt] = (df[col] < threshold).astype(np.int8).to_numpy()

    demo_cols: dict[str, pd.Series] = {}
    for src_col, canon_name in _DEMO_CANDIDATES.items():
        if src_col in df.columns and canon_name not in demo_cols:
            demo_cols[canon_name] = df[src_col]
    demo = pd.DataFrame(demo_cols)

    is_complete_case = ~np.isnan(X).any(axis=1)

    splits = {"student": KFold(n_splits=5, shuffle=True, random_state=42)}

    return Dataset(
        name="D4",
        X=X,
        y_dict=y_dict,
        demo=demo,
        feature_names=feature_cols,
        splits=splits,
        demo_axes={"primary": "race"},
        is_complete_case=is_complete_case,
    )


def _build_d4_cross_spell(
    *,
    name: str,
    target_grade: str,
    feature_grades: tuple[str, ...],
    expected_n_min: int,
    expected_n_max: int,
) -> Dataset:
    """Eligibility: G1 spelling SS < 100. Target: <target_grade>_Spell_SS >= 100."""
    if not _CSV_PATH.exists():
        raise FileNotFoundError(f"D4.csv not found at {_CSV_PATH}.")
    df_raw = pd.read_csv(_CSV_PATH)
    # shipped per-cohort file: already only this cohort, no `cohort` column
    df_raw = df_raw.reset_index(drop=True)

    g1_col = "G1_WRAT3Spell_SS"
    target_col = f"{target_grade}_WRAT3Spell_SS"
    for c in (g1_col, target_col):
        if c not in df_raw.columns:
            raise RuntimeError(f"D4 CSV missing required column: {c}")

    paired = df_raw[g1_col].notna() & df_raw[target_col].notna()
    df = df_raw.loc[paired].reset_index(drop=True)
    eligible = df[g1_col] < 100
    df = df.loc[eligible].reset_index(drop=True)
    if not (expected_n_min <= len(df) <= expected_n_max):
        raise RuntimeError(
            f"{name} eligible cohort {len(df)} outside "
            f"[{expected_n_min}, {expected_n_max}]. Did the CSV change?"
        )

    prefixes = tuple(f"{g}_" for g in feature_grades)
    feature_cols = sorted([c for c in df.columns if c.startswith(prefixes)])
    if len(feature_cols) < 10:
        raise RuntimeError(
            f"{name} has only {len(feature_cols)} feature columns "
            f"(grades {feature_grades})."
        )
    X = df[feature_cols].astype(np.float64).to_numpy()

    y = (df[target_col] >= 100).astype(np.int8).to_numpy()
    y_dict = {"crossed_spell": y}

    demo_cols: dict[str, pd.Series] = {}
    for src_col, canon_name in _DEMO_CANDIDATES.items():
        if src_col in df.columns and canon_name not in demo_cols:
            demo_cols[canon_name] = df[src_col]
    demo = pd.DataFrame(demo_cols)

    is_complete_case = ~np.isnan(X).any(axis=1)
    splits = {"student": KFold(n_splits=5, shuffle=True, random_state=42)}

    return Dataset(
        name=name,
        X=X,
        y_dict=y_dict,
        demo=demo,
        feature_names=feature_cols,
        splits=splits,
        demo_axes={"primary": "race"},
        is_complete_case=is_complete_case,
    )


@register_dataset("D4-CROSS-Spell-G2")
def load_d4_cross_spell_g2() -> Dataset:
    """G2 outcome with G1 features (1-year horizon)."""
    return _build_d4_cross_spell(
        name="D4-CROSS-Spell-G2",
        target_grade="G2",
        feature_grades=("G1",),
        expected_n_min=50, expected_n_max=110,
    )


@register_dataset("D4-CROSS-Spell")
def load_d4_cross_spell() -> Dataset:
    """G3 outcome with G1 features."""
    return _build_d4_cross_spell(
        name="D4-CROSS-Spell",
        target_grade="G3",
        feature_grades=("G1",),
        expected_n_min=40, expected_n_max=100,
    )


@register_dataset("D4-CROSS-Spell-G3-rich")
def load_d4_cross_spell_g3_rich() -> Dataset:
    """G3 outcome with G1+G2 features (richer predictor window, same horizon)."""
    return _build_d4_cross_spell(
        name="D4-CROSS-Spell-G3-rich",
        target_grade="G3",
        feature_grades=("G1", "G2"),
        expected_n_min=40, expected_n_max=100,
    )


@register_dataset("D4-CROSS-Spell-G4")
def load_d4_cross_spell_g4() -> Dataset:
    """G4 outcome with G1 features (3-year horizon, same predictor window)."""
    return _build_d4_cross_spell(
        name="D4-CROSS-Spell-G4",
        target_grade="G4",
        feature_grades=("G1",),
        expected_n_min=30, expected_n_max=80,
    )


@register_dataset("D4-CROSS-Spell-G4-rich")
def load_d4_cross_spell_g4_rich() -> Dataset:
    """G4 outcome with G1+G2+G3 features (longest horizon, richest features)."""
    return _build_d4_cross_spell(
        name="D4-CROSS-Spell-G4-rich",
        target_grade="G4",
        feature_grades=("G1", "G2", "G3"),
        expected_n_min=30, expected_n_max=80,
    )
