"""D3-G1-2 loader — Grades 1-2 multi-source cohort.

Joins demographics, DIBELS, FAST STAR, and iReady CSVs on student id;
exposes fall-of-year predictors, binary at-risk targets derived from
spring outcomes, per-student demographics (primary axis: ELL), and a
school-aware GroupKFold splitter.
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
_CSV_PATH = _ROOT.parent / "datasets" / "D3" / "D3_G12.csv"


_DIBELS_FEATURE_COLS = ["LNF", "PSF", "NWF_Letter", "NWF_Word", "ORF_WC"]

_FAST_PM1_FEATURE_COLS = ["UnifiedScore", "FAST_Equivalent_Score", "PR", "NCE"]

_IREADY_FALL_FEATURE_COLS = [
    "Overall Scale Score",
    "Phonological Awareness Scale Score",
    "Phonics Scale Score",
    "Vocabulary Scale Score",
]

_FEATURE_NAMES = [
    "LNF",
    "PSF",
    "NWF_Letter",
    "NWF_Word",
    "ORF_WC",
    "UnifiedScore",
    "FAST_Equivalent_Score",
    "PR",
    "NCE",
    "Overall_Scale_Score",
    "Phonological_Awareness_Scale_Score",
    "Phonics_Scale_Score",
    "Vocabulary_Scale_Score",
]

_IREADY_SPRING_TARGET_COL = "Overall Relative Placement"
_IREADY_ON_OR_ABOVE = frozenset({"Early On Grade Level", "Mid or Above Grade Level"})

_FAST_SPRING_TARGET_COL = "AchievementLevel"
_FAST_BELOW_LEVEL_3 = frozenset({"Level 1", "Level 2"})

_FAST_LEVEL3_COL = "Level3_or_Above"


@register_dataset("D3-G1-2")
def load_d3_g12() -> Dataset:
    """Load the D3 Grades 1-2 multi-source dataset."""
    # The release ships the already-joined panel rather than the four source
    # extracts: fall features LEFT-joined onto the demographic roster, spring
    # outcomes INNER-joined, iReady scale-score columns already normalised to
    # underscore names. Reading it directly reproduces the same frame.
    if not _CSV_PATH.exists():
        raise FileNotFoundError(
            f"D3_G12.csv not found at {_CSV_PATH}."
        )
    df = pd.read_csv(_CSV_PATH).reset_index(drop=True)

    # The released file names the two spring outcome columns after their source
    # instrument; map them back to the names the target derivation below uses.
    df = df.rename(columns={
        "FAST_PM3_AchievementLevel": _FAST_SPRING_TARGET_COL,
        "iReady_Spring_Placement": _IREADY_SPRING_TARGET_COL,
    })

    missing = [c for c in _FEATURE_NAMES if c not in df.columns]
    if missing:
        raise RuntimeError(f"D3_G12.csv missing feature columns: {missing}")

    X = df[_FEATURE_NAMES].astype(np.float64).to_numpy()

    y_fast = df[_FAST_SPRING_TARGET_COL].isin(_FAST_BELOW_LEVEL_3).astype(np.int8).to_numpy()
    y_iready = (~df[_IREADY_SPRING_TARGET_COL].isin(_IREADY_ON_OR_ABOVE)).astype(np.int8).to_numpy()

    y_dict = {
        "below_level_3_fast": y_fast,
        "below_grade_iready": y_iready,
    }

    # ELL: LEP_CODE != 'ZZ' → 1 (ELL), else 0.
    if "ELL" not in df.columns:
        df["ELL"] = (df["LEP_CODE"] != "ZZ").astype(np.int8)

    demo = df[["ELL", "SCHOOL_ID", "GRADE", "GENDER",
               "SINGLE_ETHNICITY", "ESE", "S504", "GIFTED",
               "PRIM_HOME_LANG"]].rename(columns={"SCHOOL_ID": "school_id"}).copy()

    school_groups = demo["school_id"].to_numpy()
    splits = {
        "student": KFold(n_splits=5, shuffle=True, random_state=42),
        "school": GroupKFoldWithGroups(GroupKFold(n_splits=5), school_groups),
    }

    is_complete_case = ~np.isnan(X).any(axis=1)

    return Dataset(
        name="D3-G1-2",
        X=X,
        y_dict=y_dict,
        demo=demo,
        feature_names=_FEATURE_NAMES,
        splits=splits,
        demo_axes={"primary": "ELL"},
        is_complete_case=is_complete_case,
    )


@register_dataset("D3-G1-2-CROSS")
def load_d3_g12_cross() -> Dataset:
    """D3-G1-2 cohort restricted to students who started below Level 3 at PM1.

    Target `crossed_proficiency` = 1 iff PM3 Level3_or_Above == "Yes".

    Not available from the public release. Restricting the cohort to students
    who began below Level 3 needs the PM1 achievement level, and the released
    D3 panel carries only the PM1 *scores* (UnifiedScore, FAST_Equivalent_Score,
    PR, NCE) plus the PM3 level. Deriving the target without that restriction
    would silently yield the full 2,088-student cohort rather than the 1,582
    who started below proficiency, so this loader raises instead.
    """
    raise FileNotFoundError(
        "D3-G1-2-CROSS requires the PM1 achievement-level column "
        f"('{_FAST_LEVEL3_COL}'), which is not part of the released D3 panel at "
        f"{_CSV_PATH}. Use 'D3-G1-2' for the released Grades 1-2 cohort."
    )


def _load_d3_g12_cross_from_source(_DATA_DIR, _DEMO_FILE, _DIBELS_FILE,
                                   _FAST_FILE, _IREADY_FILE) -> Dataset:
    """Original four-extract construction, retained for the source-data setting."""
    df_demo = pd.read_csv(_DATA_DIR / _DEMO_FILE)
    df_dibels = pd.read_csv(_DATA_DIR / _DIBELS_FILE)
    df_fast = pd.read_csv(_DATA_DIR / _FAST_FILE)
    df_iready = pd.read_csv(_DATA_DIR / _IREADY_FILE)

    df_dibels = df_dibels.rename(columns={"Coded ID": "CODED_ID"})
    df_iready = df_iready.rename(columns={"Coded ID": "CODED_ID"})

    dibels_fall = df_dibels[["CODED_ID"] + _DIBELS_FEATURE_COLS].copy()

    fast_pm1 = (
        df_fast[df_fast["SW_Name"] == "PM1"]
        .sort_values("Assess_Num")
        .groupby("CODED_ID", as_index=False)
        .last()[["CODED_ID"] + _FAST_PM1_FEATURE_COLS + [_FAST_LEVEL3_COL]]
        .rename(columns={_FAST_LEVEL3_COL: "Level3_or_Above_PM1"})
    )

    fast_pm3 = (
        df_fast[df_fast["SW_Name"] == "PM3"]
        .sort_values("Assess_Num")
        .groupby("CODED_ID", as_index=False)
        .last()[["CODED_ID", _FAST_LEVEL3_COL]]
        .rename(columns={_FAST_LEVEL3_COL: "Level3_or_Above_PM3"})
    )

    iready_fall = (
        df_iready[df_iready["Norming Window"].str.startswith("Fall")]
        .sort_values("Start Date")
        .groupby("CODED_ID", as_index=False)
        .last()[["CODED_ID"] + _IREADY_FALL_FEATURE_COLS]
    )

    df = (
        df_demo
        .merge(dibels_fall, on="CODED_ID", how="left")
        .merge(fast_pm1, on="CODED_ID", how="inner")
        .merge(iready_fall, on="CODED_ID", how="left")
        .merge(fast_pm3, on="CODED_ID", how="inner")
    )

    df = df[df["Level3_or_Above_PM1"] == "No"].reset_index(drop=True)

    iready_rename = {
        "Overall Scale Score": "Overall_Scale_Score",
        "Phonological Awareness Scale Score": "Phonological_Awareness_Scale_Score",
        "Phonics Scale Score": "Phonics_Scale_Score",
        "Vocabulary Scale Score": "Vocabulary_Scale_Score",
    }
    df = df.rename(columns=iready_rename)

    X = df[_FEATURE_NAMES].astype(np.float64).to_numpy()

    labeled_ids = set(df["CODED_ID"].astype(np.int64).tolist())
    fast_pm1_features_only = (
        fast_pm1.drop(columns=["Level3_or_Above_PM1"])
    )
    df_unlab = (
        df_demo
        .merge(dibels_fall, on="CODED_ID", how="left")
        .merge(fast_pm1_features_only, on="CODED_ID", how="left")
        .merge(iready_fall, on="CODED_ID", how="left")
    )
    df_unlab = df_unlab[~df_unlab["CODED_ID"].astype(np.int64).isin(labeled_ids)].copy()
    df_unlab = df_unlab.rename(columns=iready_rename)
    feat_block = df_unlab[_FEATURE_NAMES]
    df_unlab = df_unlab[~feat_block.isna().all(axis=1)].reset_index(drop=True)
    X_unlabeled = df_unlab[_FEATURE_NAMES].astype(np.float64).to_numpy()

    y_cross = (df["Level3_or_Above_PM3"] == "Yes").astype(np.int8).to_numpy()
    y_dict = {"crossed_proficiency": y_cross}

    if "ELL" not in df.columns:
        df["ELL"] = (df["LEP_CODE"] != "ZZ").astype(np.int8)
    demo = df[["ELL", "SCHOOL_ID", "GRADE", "GENDER",
               "SINGLE_ETHNICITY", "ESE", "S504", "GIFTED",
               "PRIM_HOME_LANG"]].rename(columns={"SCHOOL_ID": "school_id"}).copy()

    school_groups = demo["school_id"].to_numpy()
    splits = {
        "student": KFold(n_splits=5, shuffle=True, random_state=42),
        "school": GroupKFoldWithGroups(GroupKFold(n_splits=5), school_groups),
    }

    is_complete_case = ~np.isnan(X).any(axis=1)

    return Dataset(
        name="D3-G1-2-CROSS",
        X=X,
        y_dict=y_dict,
        demo=demo,
        feature_names=_FEATURE_NAMES,
        splits=splits,
        demo_axes={"primary": "ELL"},
        is_complete_case=is_complete_case,
        X_unlabeled=X_unlabeled,
        X_unlabeled_for_pretrain=False,
    )
