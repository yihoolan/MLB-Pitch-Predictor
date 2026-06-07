from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from utils.features.feature_names import (
    BATTER_OUTCOME_COLUMNS,
    BATTER_USAGE_COLUMNS,
    CANDIDATE_GAME_STATE_COLUMNS,
    PITCHER_OUTCOME_COLUMNS,
    PITCHER_USAGE_COLUMNS,
)

BASE_COLS = ["on_1b", "on_2b", "on_3b"]
CATEGORICAL_COLS = {"stand", "p_throws"}


def binarize_bases(df: pd.DataFrame) -> pd.DataFrame:
    """Runner ID present on base → 1, empty base → 0."""
    out = df.copy()
    for col in BASE_COLS:
        if col in out.columns:
            out[col] = out[col].notna().astype(float)
    return out


class UsageImputer:
    """Zero-vs-global-prior imputation for pitch_usage and bat_pitch_usage columns."""

    def __init__(
        self,
        usage_cols: list[str],
        *,
        stratify_col: str | None = None,
    ) -> None:
        self.usage_cols = usage_cols
        self.stratify_col = stratify_col
        self.global_prior_: pd.Series | None = None
        self.stratified_prior_: dict[object, pd.Series] | None = None

    def fit(self, df: pd.DataFrame) -> UsageImputer:
        present = [c for c in self.usage_cols if c in df.columns]
        if not present:
            self.global_prior_ = pd.Series(dtype=float)
            return self

        self.global_prior_ = df[present].median()

        if self.stratify_col and self.stratify_col in df.columns:
            self.stratified_prior_ = {
                key: group[present].median() for key, group in df.groupby(self.stratify_col, dropna=False)
            }
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        present = [c for c in self.usage_cols if c in out.columns]
        if not present or self.global_prior_ is None:
            return out

        has_row = out[present].notna().any(axis=1)

        for col in present:
            missing = out[col].isna()
            out.loc[has_row & missing, col] = 0.0

        no_row = ~has_row
        if no_row.any():
            if self.stratified_prior_ and self.stratify_col in out.columns:
                for key, prior in self.stratified_prior_.items():
                    mask = no_row & (out[self.stratify_col] == key)
                    out.loc[mask, present] = out.loc[mask, present].fillna(prior)
            out.loc[no_row, present] = out.loc[no_row, present].fillna(self.global_prior_)

        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


def outcome_pca(
    df: pd.DataFrame,
    side: Literal["pitcher", "batter"],
    n_components: int = 5,
    *,
    random_state: int = 42,
) -> tuple[np.ndarray, PCA, list[str], StandardScaler, SimpleImputer]:
    """Exploratory only (nb04 Section 1); not used in MODEL_FEATURES or production pipeline."""
    if side == "pitcher":
        cols = [c for c in PITCHER_OUTCOME_COLUMNS if c in df.columns]
        prefix = "pit"
    else:
        cols = [c for c in BATTER_OUTCOME_COLUMNS if c in df.columns]
        prefix = "bat"

    if not cols:
        empty = np.zeros((len(df), 0))
        return empty, PCA(), [], StandardScaler(), SimpleImputer()

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X = scaler.fit_transform(imputer.fit_transform(df[cols]))

    n_components = min(n_components, X.shape[1], X.shape[0])
    pca = PCA(n_components=n_components, random_state=random_state)
    components = pca.fit_transform(X)

    labels = [f"PC{i + 1}_{prefix}" for i in range(components.shape[1])]
    return components, pca, labels, scaler, imputer


def build_feature_matrix(
    df: pd.DataFrame,
    candidate_cols: list[str],
    *,
    fit: bool = True,
    preprocessor: ColumnTransformer | None = None,
    usage_imputers: dict[str, UsageImputer] | None = None,
    pca_components: np.ndarray | None = None,
    pca_labels: list[str] | None = None,
) -> tuple[np.ndarray, list[str], ColumnTransformer | None]:
    """Assemble a numeric feature matrix for a candidate column set.

    Production training uses MODEL_FEATURES (Set B). Pass other CANDIDATE_SET_* lists
    for comparison experiments (Sets A–E).
    """
    cols = [c for c in candidate_cols if c in df.columns]
    work = binarize_bases(df)

    ### Usage imputation when usage cols are in the candidate set
    if usage_imputers is None:
        usage_imputers = {}
        if any(c in cols for c in PITCHER_USAGE_COLUMNS):
            usage_imputers["pitcher"] = UsageImputer(PITCHER_USAGE_COLUMNS, stratify_col="p_throws")
        if any(c in cols for c in BATTER_USAGE_COLUMNS):
            usage_imputers["batter"] = UsageImputer(BATTER_USAGE_COLUMNS, stratify_col="stand")

    for imputer in usage_imputers.values():
        if fit:
            work = imputer.fit_transform(work)
        else:
            work = imputer.transform(work)

    cat_cols = [c for c in cols if c in CATEGORICAL_COLS or work[c].dtype == object]
    num_cols = [c for c in cols if c not in cat_cols]

    if preprocessor is None:
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "cat",
                    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                    cat_cols,
                ),
                ("num", SimpleImputer(strategy="median"), num_cols),
            ],
            remainder="drop",
        )

    if fit:
        X = preprocessor.fit_transform(work[cols])
    else:
        X = preprocessor.transform(work[cols])

    feature_names = cat_cols + num_cols

    if pca_components is not None and pca_components.shape[1] > 0:
        X = np.hstack([X, pca_components])
        feature_names = feature_names + (pca_labels or [])

    return X, feature_names, preprocessor


def feature_group(name: str) -> str:
    """Map a feature name to its EDA tier for importance plots.

    Outcome and PCA groups remain for nb04 plots and rejected Sets C–E.
    """
    if name.startswith("PC") and (name.endswith("_pit") or name.endswith("_bat")):
        side = "pitcher" if name.endswith("_pit") else "batter"
        return f"{side} outcome PCA"
    if name in CANDIDATE_GAME_STATE_COLUMNS:
        return "Game state"
    if name in PITCHER_USAGE_COLUMNS:
        return "Pitcher usage"
    if name in BATTER_USAGE_COLUMNS:
        return "Batter usage"
    if name in PITCHER_OUTCOME_COLUMNS:
        return "Pitcher outcome"
    if name in BATTER_OUTCOME_COLUMNS:
        return "Batter outcome"
    return "Other"
