from __future__ import annotations

import pandas as pd
from pybaseball import statcast_batter_pitch_arsenal, statcast_pitcher_arsenal_stats

PIVOT_ID_COLS = [
    "last_name, first_name",
    "player_id",
    "team_name_alt",
    "pitch_type",
    "pitch_name",
]


def _pivot_arsenal_stats(
    table: pd.DataFrame,
    *,
    id_col: str,
    col_prefix: str = "",
) -> pd.DataFrame:
    stat_cols = [c for c in table.columns if c not in PIVOT_ID_COLS]
    wide = table.pivot_table(
        index="player_id",
        columns="pitch_type",
        values=stat_cols,
        aggfunc="first",
    )
    wide.columns = [f"{col_prefix}{stat}_{pt}" for stat, pt in wide.columns]
    return wide.reset_index().rename(columns={"player_id": id_col})


def enrich_statcast(
    df: pd.DataFrame,
    prior_year: int,
    min_pa: int = 25,
) -> pd.DataFrame:
    """Pull prior-year pitcher/batter arsenal stats, pivot wide, merge onto pitch rows."""
    pit_outcomes = statcast_pitcher_arsenal_stats(prior_year, minPA=min_pa)
    pit_outcomes_w = _pivot_arsenal_stats(pit_outcomes, id_col="pitcher")

    bat_vs_pitch = statcast_batter_pitch_arsenal(prior_year, minPA=min_pa)
    bat_vs_pitch_w = _pivot_arsenal_stats(bat_vs_pitch, id_col="batter", col_prefix="bat_")

    return df.merge(pit_outcomes_w, on="pitcher", how="left").merge(bat_vs_pitch_w, on="batter", how="left")
