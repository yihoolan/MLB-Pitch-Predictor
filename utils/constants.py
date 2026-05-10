### Columns from the raw Statcast frame that are safe to use as training features.
### training can only be done on pre-pitch context. Therefore, post-pitch features like spin rates
### must be removed to prevent data leakage
### Source: https://baseballsavant.mlb.com/csv-docs
TRAINABLE_COLUMNS: list[str] = [
    # Game state
    "balls",
    "strikes",
    "outs_when_up",
    "inning",
    "inning_topbot",
    "on_1b",
    "on_2b",
    "on_3b",
    "home_score",
    "away_score",
    "bat_score",
    "fld_score",
    "pitch_number",
    "at_bat_number",
    # Identifiers / handedness
    "pitcher",
    "batter",
    "stand",
    "p_throws",
    "home_team",
    "away_team",
    "game_pk",
    "game_date",
    "game_year",
    # Defensive setup (determined before the pitch)
    "if_fielding_alignment",
    "of_fielding_alignment",
    "fielder_2",
    "fielder_3",
    "fielder_4",
    "fielder_5",
    "fielder_6",
    "fielder_7",
    "fielder_8",
    "fielder_9",
    # Strike-zone geometry (batter-defined, not pitch-defined)
    "sz_top",
    "sz_bot",
    # Game / season metadata
    "n_thruorder_pitcher",
    "n_priorpa_thisgame_player_at_bat",
    "pitcher_days_since_prev_game",
    "batter_days_since_prev_game",
    "age_pit",
    "age_bat",
]

LABEL_COLUMN: str = "pitch_type"
