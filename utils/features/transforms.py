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
