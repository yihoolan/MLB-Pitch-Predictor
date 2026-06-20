import math

import pandas as pd
import pytest

from utils.transforms import UsageImputer

COLS = ["pitch_usage_FF", "pitch_usage_SL", "pitch_usage_CH"]


def _df(*rows):
    return pd.DataFrame(rows, columns=COLS)


def test_full_row_passthrough():
    df = _df({"pitch_usage_FF": 0.5, "pitch_usage_SL": 0.3, "pitch_usage_CH": 0.2})
    result = UsageImputer(COLS).fit_transform(df)
    assert result["pitch_usage_FF"].iloc[0] == pytest.approx(0.5)
    assert result["pitch_usage_SL"].iloc[0] == pytest.approx(0.3)
    assert result["pitch_usage_CH"].iloc[0] == pytest.approx(0.2)


def test_partial_row_fills_zeros():
    # Player has at least one usage stat → has_row=True → missing cols get 0.0, not the median.
    train = _df({"pitch_usage_FF": 0.6, "pitch_usage_SL": 0.3, "pitch_usage_CH": 0.1})
    imputer = UsageImputer(COLS)
    imputer.fit(train)

    partial = _df({"pitch_usage_FF": 0.7, "pitch_usage_SL": float("nan"), "pitch_usage_CH": float("nan")})
    result = imputer.transform(partial)

    assert result["pitch_usage_FF"].iloc[0] == pytest.approx(0.7)
    assert result["pitch_usage_SL"].iloc[0] == pytest.approx(0.0)
    assert result["pitch_usage_CH"].iloc[0] == pytest.approx(0.0)


def test_rookie_fills_global_median():
    # Player has no usage row at all → has_row=False → all cols get the fitted global median.
    train = _df(
        {"pitch_usage_FF": 0.6, "pitch_usage_SL": 0.3, "pitch_usage_CH": 0.1},
        {"pitch_usage_FF": 0.4, "pitch_usage_SL": 0.5, "pitch_usage_CH": 0.1},
    )
    imputer = UsageImputer(COLS)
    imputer.fit(train)

    rookie = _df({"pitch_usage_FF": float("nan"), "pitch_usage_SL": float("nan"), "pitch_usage_CH": float("nan")})
    result = imputer.transform(rookie)

    assert result["pitch_usage_FF"].iloc[0] == pytest.approx(0.5)   # median of [0.6, 0.4]
    assert result["pitch_usage_SL"].iloc[0] == pytest.approx(0.4)   # median of [0.3, 0.5]
    assert result["pitch_usage_CH"].iloc[0] == pytest.approx(0.1)   # median of [0.1, 0.1]
