from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Query

from app.enrichment import search_players
from app.schemas import PlayerMatch

router = APIRouter(prefix="/players", tags=["players"])


@router.get("", response_model=list[PlayerMatch])
def get_players(
    query: str = Query(..., min_length=2, description="Partial player name to search for"),
    role: Literal["pitcher", "batter"] = Query("pitcher", description="Used for display context only"),
) -> list[PlayerMatch]:
    """Return up to 10 player name matches for the given query string.

    Searches pybaseball's local Chadwick register (Statcast era players only).
    Results are sorted by most recent MLB activity.
    """
    return search_players(query, role=role)
