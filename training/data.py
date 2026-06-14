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
