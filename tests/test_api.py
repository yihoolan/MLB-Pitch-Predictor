import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from utils.feature_names import BATTER_USAGE_COLUMNS, PITCHER_USAGE_COLUMNS

from app.main import app

_EMPTY_USAGE = {col: None for col in PITCHER_USAGE_COLUMNS + BATTER_USAGE_COLUMNS}

_VALID_REQUEST = {
    "pitcher_mlbam_id": 1,
    "pitcher_name": "Test Pitcher",
    "batter_mlbam_id": 2,
    "batter_name": "Test Batter",
    "balls": 1,
    "strikes": 1,
    "outs_when_up": 1,
    "inning": 5,
    "pitch_number": 3,
    "bat_score_diff": 0,
    "on_1b": False,
    "on_2b": True,
    "on_3b": False,
    "stand": "R",
    "p_throws": "R",
}


def test_predict_base_runner_encoding():
    captured = []

    mock_registry = MagicMock()
    mock_registry.is_loaded = True
    mock_registry.version = "1"
    mock_registry.model.predict.side_effect = lambda df: captured.append(df) or np.ones((1, 10)) / 10

    with (
        patch("app.main.model_registry.load_production"),
        patch("app.main._get_arsenal_tables"),
        patch("app.routers.predict.enrich_row", return_value=(_EMPTY_USAGE, False, False)),
        patch("app.routers.predict.model_registry", mock_registry),
    ):
        with TestClient(app) as client:
            resp = client.post("/predict", json=_VALID_REQUEST)

    assert resp.status_code == 200
    df = captured[0]
    assert math.isnan(df["on_1b"].iloc[0])
    assert df["on_2b"].iloc[0] == pytest.approx(1.0)


def test_health_smoke():
    mock_registry = MagicMock()
    mock_registry.is_loaded = True
    mock_registry.version = "1"

    with (
        patch("app.main.model_registry.load_production"),
        patch("app.main._get_arsenal_tables"),
        patch("app.main.model_registry", mock_registry),
    ):
        with TestClient(app) as client:
            resp = client.get("/health")

    assert resp.status_code == 200
    assert resp.json()["model_loaded"] is True
