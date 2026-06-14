from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class GameStateRequest(BaseModel):
    pitcher_name: str = Field(..., description="Pitcher full name, e.g. 'Gerrit Cole'")
    batter_name: str = Field(..., description="Batter full name, e.g. 'Aaron Judge'")

    # Count
    balls: int = Field(..., ge=0, le=3)
    strikes: int = Field(..., ge=0, le=2)
    outs_when_up: int = Field(..., ge=0, le=2)

    # Inning / at-bat context
    inning: int = Field(..., ge=1, le=15)
    pitch_number: int = Field(..., ge=1, le=20)
    bat_score_diff: int = Field(..., ge=-20, le=20, description="Batting team score minus fielding team score")

    # Base runners (True = runner present)
    on_1b: bool = False
    on_2b: bool = False
    on_3b: bool = False

    # Handedness
    stand: Literal["L", "R"] = Field(..., description="Batter stance")
    p_throws: Literal["L", "R"] = Field(..., description="Pitcher throwing arm")


class PitchProbabilities(BaseModel):
    """Predicted probability for each Statcast pitch type, summing to 1.0."""

    probabilities: dict[str, float]
    top_pitch: str
    pitcher_name: str
    batter_name: str
    rookie_pitcher: bool = Field(False, description="True if pitcher had no prior-year stats")
    rookie_batter: bool = Field(False, description="True if batter had no prior-year stats")


class PlayerMatch(BaseModel):
    name: str
    mlbam_id: int
    throws_or_stands: str = Field(..., description="Pitching arm (L/R) for pitchers; batting stance for batters")


class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    status: str
    model_version: str | None
    model_loaded: bool
