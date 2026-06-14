"""Player name search and prior-year arsenal enrichment.

search_players() does a case-insensitive substring match against the full
player name using pybaseball's local Chadwick register, then enriches each
result with real handedness data from the MLB Stats API.

enrich_row() fetches prior-year pitch_usage stats for a pitcher/batter pair
and returns a dict of the 20 MODEL_FEATURES usage columns. The full year
tables are cached in-process so every player lookup after the first is free.
Players with no prior-year row (rookies) get NaN for all usage columns,
which the model's saved UsageImputer fills with the training-time global median.
"""

from __future__ import annotations

import datetime
from typing import Literal

import pandas as pd
import pybaseball
import requests
from pybaseball import statcast_batter_pitch_arsenal, statcast_pitcher_arsenal_stats
from pybaseball.playerid_lookup import get_lookup_table

pybaseball.cache.enable()

from app.schemas import PlayerMatch
from utils.feature_names import BATTER_USAGE_COLUMNS, PITCHER_USAGE_COLUMNS

_STATCAST_ERA_START = 2015


# ---------------------------------------------------------------------------
# Player name search
# ---------------------------------------------------------------------------


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


# Permanent cache: handedness never changes for a given player.
_handedness_cache: dict[int, str] = {}


def _fetch_handedness(mlbam_ids: list[int], role: Literal["pitcher", "batter"]) -> None:
    """Populate _handedness_cache for any IDs not already present.

    Makes a single batch request to the MLB Stats API. Falls back silently on
    network errors, leaving missing IDs absent from the cache (callers use "?").
    Switch hitters (batSide == "S") are mapped to "R" since the model only
    supports L/R.
    """
    missing = [i for i in mlbam_ids if i not in _handedness_cache]
    if not missing:
        return
    # No fields filter — some MLB Stats API deployments ignore or truncate it.
    url = (
        "https://statsapi.mlb.com/api/v1/people"
        f"?personIds={','.join(str(i) for i in missing)}"
    )
    try:
        for p in requests.get(url, timeout=5).json().get("people", []):
            field = "pitchHand" if role == "pitcher" else "batSide"
            code = (p.get(field) or {}).get("code", "?")
            _handedness_cache[p["id"]] = "R" if code == "S" else code
    except Exception:
        pass


def search_players(
    query: str,
    role: Literal["pitcher", "batter"] = "pitcher",
    max_results: int = 10,
) -> list[PlayerMatch]:
    """Fuzzy player name search against the Chadwick register.

    Splits query into words and keeps rows where every word appears somewhere
    in the player's full name. Returns up to max_results matches, sorted by
    most recent activity first, with real handedness populated from the MLB
    Stats API (cached permanently after the first lookup per player).

    role is passed through to PlayerMatch; it does not filter the register
    (a pitcher/batter distinction isn't in the Chadwick register).
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
    mlbam_ids = [int(row.key_mlbam) for row in matches.itertuples()]

    _fetch_handedness(mlbam_ids, role)

    return [
        PlayerMatch(
            name=f"{row.name_first.title()} {row.name_last.title()}",
            mlbam_id=int(row.key_mlbam),
            throws_or_stands=_handedness_cache.get(int(row.key_mlbam), "?"),
        )
        for row in matches.itertuples()
    ]


# ---------------------------------------------------------------------------
# Arsenal enrichment
# ---------------------------------------------------------------------------

# Cache keyed by year; each entry holds the full pitcher/batter usage tables
# for that prior year so we only pay the pybaseball download cost once.
_arsenal_cache: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}

# Cache keyed by (pitcher_mlbam, batter_mlbam, year); player stats for a given
# year are fixed, so no TTL is needed. Avoids repeated DataFrame filters on
# every POST /predict (most useful for the What-If explorer).
_enrich_cache: dict[tuple[int, int, int], tuple[dict, bool, bool]] = {}


def _current_prior_year() -> int:
    return datetime.datetime.now().year - 1


def _get_arsenal_tables(year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (pitcher_wide, batter_wide) usage tables for year, from cache or pybaseball."""
    if year not in _arsenal_cache:
        pit_raw = statcast_pitcher_arsenal_stats(year, minPA=25)
        bat_raw = statcast_batter_pitch_arsenal(year, minPA=25)
        _arsenal_cache[year] = (_pivot_pitcher_usage(pit_raw), _pivot_batter_usage(bat_raw))
    return _arsenal_cache[year]


def _pivot_pitcher_usage(table: pd.DataFrame) -> pd.DataFrame:
    """Pivot long arsenal table to one row per player_id with pitch_usage_FF etc."""
    wide = table.pivot_table(index="player_id", columns="pitch_type", values="pitch_usage", aggfunc="first")
    wide.columns = [f"pitch_usage_{pt}" for pt in wide.columns]
    for col in PITCHER_USAGE_COLUMNS:
        if col not in wide.columns:
            wide[col] = float("nan")
    return wide[PITCHER_USAGE_COLUMNS].reset_index()


def _pivot_batter_usage(table: pd.DataFrame) -> pd.DataFrame:
    """Pivot long batter table to one row per player_id with bat_pitch_usage_FF etc."""
    wide = table.pivot_table(index="player_id", columns="pitch_type", values="pitch_usage", aggfunc="first")
    wide.columns = [f"bat_pitch_usage_{pt}" for pt in wide.columns]
    for col in BATTER_USAGE_COLUMNS:
        if col not in wide.columns:
            wide[col] = float("nan")
    return wide[BATTER_USAGE_COLUMNS].reset_index()


def enrich_row(
    pitcher_mlbam: int,
    batter_mlbam: int,
    prior_year: int | None = None,
) -> tuple[dict[str, float | None], bool, bool]:
    """Return usage feature dict for a pitcher/batter pair for inference.

    Fetches the full prior-year arsenal tables (cached after first call) and
    looks up each player by MLBAM ID. Results are cached per (pitcher, batter,
    year) so repeated calls with the same pair (e.g. What-If explorer) are free.

    Players with no prior-year row (rookies) get NaN for all their usage columns;
    the model's saved UsageImputer handles those with the training-time stratified
    global median.

    Returns:
        usage_dict     — dict mapping all 20 usage column names to float or NaN
        rookie_pitcher — True if pitcher had no prior-year stats
        rookie_batter  — True if batter had no prior-year stats
    """
    year = prior_year or _current_prior_year()
    key = (pitcher_mlbam, batter_mlbam, year)
    if key in _enrich_cache:
        return _enrich_cache[key]

    pit_wide, bat_wide = _get_arsenal_tables(year)

    pit_row = pit_wide[pit_wide["player_id"] == pitcher_mlbam]
    bat_row = bat_wide[bat_wide["player_id"] == batter_mlbam]

    rookie_pitcher = pit_row.empty
    rookie_batter = bat_row.empty

    usage: dict[str, float | None] = {}
    for col in PITCHER_USAGE_COLUMNS:
        usage[col] = None if rookie_pitcher else pit_row.iloc[0][col]
    for col in BATTER_USAGE_COLUMNS:
        usage[col] = None if rookie_batter else bat_row.iloc[0][col]

    result = (usage, rookie_pitcher, rookie_batter)
    _enrich_cache[key] = result
    return result
