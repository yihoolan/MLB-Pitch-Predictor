from __future__ import annotations

import pandas as pd

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
