"""D1 dataset loader (the public early-literacy benchmark from prior work).

D1 is *not* redistributed by RealLit. To run cells against D1 you must
obtain the CSV from the original release of the prior-work benchmark
(Shangguan et al. 2024) and place it at ``dataset/D1/D1.csv`` relative
to the harness root. The harness's run_cell entry point will raise an
informative ``FileNotFoundError`` if the file is missing.

See the paper Dataset section for the labeled-cohort size, feature
count, and target list. School split is supported via the SchlID
column when present in the source CSV.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

from n26.data import register_dataset
from n26.data._splitters import GroupKFoldWithGroups
from n26.data.base import Dataset

_ROOT = Path(__file__).resolve().parents[3]
_CSV_PATH = _ROOT.parent / "datasets" / "D1" / "D1.csv"

_FEATURE_COLS = [
    "NWFcls", "NWFwrc", "ORFwc",
    "SAwrS", "SAsrS", "SAtoS",
    "RMwidRS", "RMwdaRS",
    "Gender", "Age1b", "Tx", "Tier2_N",
    "TKPctCrct", "gnrl_fid", "rcmistot",
    "Tier",
]
_TARGET_COLS = {
    "WordID": "RMwidRS_target",
    "WordAtk": "RMwdaRS_target",
}


@register_dataset("D1")
def load_d1() -> Dataset:
    if not _CSV_PATH.exists():
        raise FileNotFoundError(
            f"D1.csv not found at {_CSV_PATH}. Obtain from Shangguan et al. (2024)."
        )

    df_raw = pd.read_csv(_CSV_PATH)

    label_mask = df_raw[_TARGET_COLS["WordID"]].notna() & df_raw[_TARGET_COLS["WordAtk"]].notna()
    df = df_raw.loc[label_mask].reset_index(drop=True)
    if len(df) != 946:
        raise RuntimeError(
            f"D1 labeled cohort size {len(df)} != 946 (spec). "
            f"Did the CSV change?"
        )

    missing_features = [c for c in _FEATURE_COLS if c not in df.columns]
    if missing_features:
        raise RuntimeError(f"D1 CSV missing feature columns: {missing_features}")
    X = df[_FEATURE_COLS].astype(np.float64).to_numpy()

    # Unlabeled rows: kept for pre-training only (Shangguan et al. 2024).
    df_unlab = df_raw.loc[~label_mask].reset_index(drop=True)
    X_unlabeled = df_unlab[_FEATURE_COLS].astype(np.float64).to_numpy()

    # At-risk framing: y=1 means at-risk / non-responsive.
    y_dict = {
        "WordID": (1 - df[_TARGET_COLS["WordID"]].astype(int)).to_numpy(dtype=np.int8),
        "WordAtk": (1 - df[_TARGET_COLS["WordAtk"]].astype(int)).to_numpy(dtype=np.int8),
    }

    # D1 provides only school-aggregated demographics; expose an empty frame.
    demo = pd.DataFrame(index=range(len(df)))

    is_complete_case = ~np.isnan(X).any(axis=1)

    student_split = KFold(n_splits=5, shuffle=True, random_state=42)
    if "SchlID" in df.columns:
        school_groups = df["SchlID"].to_numpy()
        school_split = GroupKFoldWithGroups(GroupKFold(n_splits=5), school_groups)
        splits = {"student": student_split, "school": school_split}
    else:
        splits = {"student": student_split}

    return Dataset(
        name="D1",
        X=X,
        y_dict=y_dict,
        demo=demo,
        feature_names=_FEATURE_COLS,
        splits=splits,
        demo_axes={},
        is_complete_case=is_complete_case,
        X_unlabeled=X_unlabeled,
    )


@register_dataset("D1-intervention")
def load_d1_intervention() -> Dataset:
    """D1 subset restricted to intervention (Tx=1) students.

    The fine-tune cohort is the students with Tx=1 *and* both targets
    non-NaN. The pre-training corpus is everyone *not* in that cohort —
    i.e. control-labeled + Tx-NaN + label-NaN rows — so MaskMLP still sees
    the full source CSV at pre-training time.
    """
    if not _CSV_PATH.exists():
        raise FileNotFoundError(
            f"D1.csv not found at {_CSV_PATH}. Obtain from Shangguan et al. (2024)."
        )

    df_raw = pd.read_csv(_CSV_PATH)
    label_mask = df_raw[_TARGET_COLS["WordID"]].notna() & df_raw[_TARGET_COLS["WordAtk"]].notna()
    intn_mask = label_mask & (df_raw["Tx"] == 1)

    df = df_raw.loc[intn_mask].reset_index(drop=True)
    if len(df) != 429:
        raise RuntimeError(
            f"D1-intervention cohort size {len(df)} != 429 (spec). "
            f"Did the CSV change?"
        )

    missing_features = [c for c in _FEATURE_COLS if c not in df.columns]
    if missing_features:
        raise RuntimeError(f"D1 CSV missing feature columns: {missing_features}")
    X = df[_FEATURE_COLS].astype(np.float64).to_numpy()

    df_unlab = df_raw.loc[~intn_mask].reset_index(drop=True)
    X_unlabeled = df_unlab[_FEATURE_COLS].astype(np.float64).to_numpy()

    y_dict = {
        "WordID": (1 - df[_TARGET_COLS["WordID"]].astype(int)).to_numpy(dtype=np.int8),
        "WordAtk": (1 - df[_TARGET_COLS["WordAtk"]].astype(int)).to_numpy(dtype=np.int8),
    }

    demo = pd.DataFrame(index=range(len(df)))
    is_complete_case = ~np.isnan(X).any(axis=1)

    student_split = KFold(n_splits=5, shuffle=True, random_state=42)
    if "SchlID" in df.columns:
        school_groups = df["SchlID"].to_numpy()
        school_split = GroupKFoldWithGroups(GroupKFold(n_splits=5), school_groups)
        splits = {"student": student_split, "school": school_split}
    else:
        splits = {"student": student_split}

    return Dataset(
        name="D1-intervention",
        X=X,
        y_dict=y_dict,
        demo=demo,
        feature_names=_FEATURE_COLS,
        splits=splits,
        demo_axes={},
        is_complete_case=is_complete_case,
        X_unlabeled=X_unlabeled,
    )
