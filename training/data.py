from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from pybaseball import statcast
from sklearn.impute import SimpleImputer

from utils.enrichment import enrich_statcast
from utils.feature_names import (
    BATTER_USAGE_COLUMNS,
    LABEL_COLUMN,
    MODEL_FEATURES,
    PITCH_TYPES,
    PITCHER_USAGE_COLUMNS,
)
from utils.transforms import UsageImputer, binarize_bases

_CATEGORICAL_COLS = ["stand", "p_throws"]

### Fixed label encoding — index matches LightGBM's internal class ordering
LABEL_ENCODER: dict[str, int] = {pt: i for i, pt in enumerate(PITCH_TYPES)}
LABEL_DECODER: dict[int, str] = {i: pt for pt, i in LABEL_ENCODER.items()}


def load_season(year: int, start_month: int = 4, end_month: int = 9) -> pd.DataFrame:
    """Pull Statcast pitches for a date range and enrich with prior-year arsenal stats.

    Each year carries its own prior: 2023 data → 2022 arsenal, 2024 → 2023, etc.
    """
    end_day = (pd.Timestamp(year, end_month, 1) + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d")
    raw = statcast(f"{year}-{start_month:02d}-01", end_day)
    return enrich_statcast(raw, prior_year=year - 1)


def _preprocess(
    df: pd.DataFrame,
    feature_cols: list[str],
    cat_cols: list[str],
    num_cols: list[str],
    *,
    pitcher_imp: UsageImputer,
    batter_imp: UsageImputer,
    num_imp: SimpleImputer,
    categories: dict[str, list],
    fit: bool,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Apply the full preprocessing chain to one split.

    When fit=True, imputers are fitted on df and categories is populated in place.
    When fit=False, previously fitted imputers and categories are applied.
    """
    work = binarize_bases(df)

    if fit:
        work = pitcher_imp.fit_transform(work)
        work = batter_imp.fit_transform(work)
        imputed_num = num_imp.fit_transform(work[num_cols])
        for col in cat_cols:
            categories[col] = sorted(work[col].dropna().unique().tolist())
    else:
        work = pitcher_imp.transform(work)
        work = batter_imp.transform(work)
        imputed_num = num_imp.transform(work[num_cols])

    work = work.copy()
    work[num_cols] = imputed_num

    for col in cat_cols:
        work[col] = pd.Categorical(work[col], categories=categories[col])

    X = work[feature_cols].copy()
    y = work[LABEL_COLUMN].map(LABEL_ENCODER).to_numpy()
    return X, y


def build_lgb_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame | None = None,
) -> tuple[lgb.Dataset, lgb.Dataset, pd.DataFrame, np.ndarray, list[str]]:
    """Preprocess all splits and return (ds_train, ds_val, X_test, y_test, feature_names).

    All imputers are fit exclusively on train_df and applied to val and test.
    If test_df is None, val_df is reused as the test set (incremental mode).
    """
    if test_df is None:
        test_df = val_df

    # Drop rows whose pitch type isn't in PITCH_TYPES; .map() would silently
    # produce NaN labels for unknown codes (EP, FA, SC, PO, …), which then
    # causes sklearn to blow up when computing metrics.
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        n_unknown = (~df[LABEL_COLUMN].isin(PITCH_TYPES)).sum()
        if n_unknown:
            print(f"  [data] dropping {n_unknown} {name} rows with unknown pitch types")
    train_df = train_df[train_df[LABEL_COLUMN].isin(PITCH_TYPES)]
    val_df = val_df[val_df[LABEL_COLUMN].isin(PITCH_TYPES)]
    test_df = test_df[test_df[LABEL_COLUMN].isin(PITCH_TYPES)]

    feature_cols = [c for c in MODEL_FEATURES if c in train_df.columns]
    cat_cols = [c for c in _CATEGORICAL_COLS if c in feature_cols]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    pitcher_imp = UsageImputer(PITCHER_USAGE_COLUMNS, stratify_col="p_throws")
    batter_imp = UsageImputer(BATTER_USAGE_COLUMNS, stratify_col="stand")
    num_imp = SimpleImputer(strategy="median")
    categories: dict[str, list] = {}

    shared = dict(
        feature_cols=feature_cols,
        cat_cols=cat_cols,
        num_cols=num_cols,
        pitcher_imp=pitcher_imp,
        batter_imp=batter_imp,
        num_imp=num_imp,
        categories=categories,
    )

    X_train, y_train = _preprocess(train_df, **shared, fit=True)
    X_val, y_val = _preprocess(val_df, **shared, fit=False)
    X_test, y_test = _preprocess(test_df, **shared, fit=False)

    ds_train = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols, free_raw_data=False)
    ds_val = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_cols, reference=ds_train, free_raw_data=False)

    return ds_train, ds_val, X_test, y_test, feature_cols
