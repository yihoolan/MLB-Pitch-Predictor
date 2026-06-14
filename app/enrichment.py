"""Player name search using pybaseball's local Chadwick register.

search_players() does a case-insensitive substring match against the full
player name and returns the closest matches, filtered to players active in
the modern Statcast era (2015+).
"""

from __future__ import annotations

from typing import Literal

import pandas as pd
from pybaseball.playerid_lookup import get_lookup_table

from app.schemas import PlayerMatch

_STATCAST_ERA_START = 2015


def _load_register() -> pd.DataFrame:
    """Return the Chadwick register filtered to Statcast-era players with MLBAM IDs."""
    df = get_lookup_table()
    df = df[df["key_mlbam"].notna() & (df["mlb_played_last"] >= _STATCAST_ERA_START)].copy()
    df["key_mlbam"] = df["key_mlbam"].astype(int)
    df["full_name"] = (df["name_first"] + " " + df["name_last"]).str.lower()
    return df


_register: pd.DataFrame | None = None


def _get_register() -> pd.DataFrame:
    global _register
    if _register is None:
        _register = _load_register()
    return _register


def search_players(
    query: str,
    role: Literal["pitcher", "batter"] = "pitcher",
    max_results: int = 10,
) -> list[PlayerMatch]:
    """Fuzzy player name search against the Chadwick register.

    Splits query into words and keeps rows where every word appears somewhere
    in the player's full name. Returns up to max_results matches, sorted by
    most recent activity first.

    role is passed through to PlayerMatch for display purposes; it does not
    filter the register (a pitcher/batter distinction isn't in the register).
    """
    register = _get_register()
    q = query.strip().lower()
    if not q:
        return []

    words = q.split()
    mask = pd.Series([True] * len(register), index=register.index)
    for word in words:
        mask &= register["full_name"].str.contains(word, regex=False)

    matches = register[mask].sort_values("mlb_played_last", ascending=False).head(max_results)

    return [
        PlayerMatch(
            name=f"{row.name_first.title()} {row.name_last.title()}",
            mlbam_id=int(row.key_mlbam),
            throws_or_stands="?",
        )
        for row in matches.itertuples()
    ]
