### Column classification for the raw Statcast frame (118 cols) and pybaseball enrichment tables.
### Source: https://baseballsavant.mlb.com/csv-docs
###
### STATCAST_DEPRECATED  – always-null legacy fields from the old PitchFX tracking system.
### STATCAST_POST_PITCH  – pitch physics and outcomes known only after the pitch is thrown;
###                        using these as model inputs would constitute data leakage.
### STATCAST_LOGISTICS   – identifiers and bookkeeping columns retained for joins / temporal
###                        splits / deduplication, but not fed to the model directly.
### STATCAST_PRE_PITCH   – valid game-state features available before the pitch is thrown.
### FANGRAPH_PRE_PITCH   – prior-year enrichment columns joined from pybaseball leaderboards
###                        (statcast_pitcher_pitch_arsenal, statcast_pitcher_arsenal_stats,
###                         statcast_batter_expected_stats, statcast_batter_pitch_arsenal).
###                        All are leak-free because they come from SEASON - 1.

### Deprecated fields
STATCAST_DEPRECATED: list[str] = [
    "spin_dir",
    "spin_rate_deprecated",
    "break_angle_deprecated",
    "break_length_deprecated",
    "tfs_deprecated",
    "tfs_zulu_deprecated",
    "sv_id",
    "umpire",
]

### Post-pitch physics stats, must be excluded to prevent data leakage
STATCAST_POST_PITCH: list[str] = [
    "release_speed",
    "effective_speed",
    "release_pos_x",
    "release_pos_z",
    "release_pos_y",
    "release_extension",
    "release_spin_rate",
    "spin_axis",
    "arm_angle",
    "pfx_x",
    "pfx_z",
    "api_break_z_with_gravity",
    "api_break_x_arm",
    "api_break_x_batter_in",
    "plate_x",
    "plate_z",
    "zone",
    "vx0",
    "vy0",
    "vz0",
    "ax",
    "ay",
    "az",
    "type",
    "events",
    "description",
    "des",
    "hit_location",
    "bb_type",
    "hc_x",
    "hc_y",
    "hit_distance_sc",
    "launch_speed",
    "launch_angle",
    "launch_speed_angle",
    "estimated_ba_using_speedangle",
    "estimated_woba_using_speedangle",
    "estimated_slg_using_speedangle",
    "woba_value",
    "woba_denom",
    "babip_value",
    "iso_value",
    "post_home_score",
    "post_away_score",
    "post_bat_score",
    "post_fld_score",
    "delta_home_win_exp",
    "delta_run_exp",
    "delta_pitcher_run_exp",
    "bat_speed",
    "swing_length",
    "attack_angle",
    "attack_direction",
    "swing_path_tilt",
    "intercept_ball_minus_batter_pos_x_inches",
    "intercept_ball_minus_batter_pos_y_inches",
    "hyper_speed",
]

### Useless pre-pitch fields (some are also post-game stats which represent leakage)
STATCAST_LOGISTICS: list[str] = [
    "game_pk",
    "game_date",
    "player_name",
    "pitch_name",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "bat_score",
    "fld_score",
    "at_bat_number",
    "fielder_3",
    "fielder_4",
    "fielder_5",
    "fielder_6",
    "fielder_7",
    "fielder_8",
    "fielder_9",
    "age_pit_legacy",
    "age_bat_legacy",
    "batter_days_until_next_game",
]

### Useful pre-pitch game-state features
STATCAST_PRE_PITCH: list[str] = [
    "balls",
    "strikes",
    "outs_when_up",
    "on_1b",
    "on_2b",
    "on_3b",
    "inning",
    "inning_topbot",
    "bat_score_diff",
    "home_score_diff",
    "home_win_exp",
    "bat_win_exp",
    "stand",
    "p_throws",
    "pitcher",
    "batter",
    "pitch_number",
    "n_thruorder_pitcher",
    "n_priorpa_thisgame_player_at_bat",
    "pitcher_days_since_prev_game",
    "pitcher_days_until_next_game",
    "batter_days_since_prev_game",
    "if_fielding_alignment",
    "of_fielding_alignment",
    "fielder_2",
    "sz_top",
    "sz_bot",
    "age_pit",
    "age_bat",
    "game_year",
    "game_type",
]

### Pitch type codes
_PITCH_TYPES: list[str] = ["FF", "SI", "FC", "SL", "CH", "CU", "FS", "KN", "ST", "SV"]

### pitcher aresenal
_ARSENAL_STATS: list[str] = [
    "run_value_per_100",
    "run_value",
    "pitches",
    "pitch_usage",
    "pa",
    "ba",
    "slg",
    "woba",
    "whiff_percent",
    "k_percent",
    "put_away",
    "est_ba",
    "est_slg",
    "est_woba",
    "hard_hit_percent",
]

### prior-year enrichment columns (all leakage-free)
FANGRAPH_PRE_PITCH: list[str] = [
    "n_ff",
    "n_si",
    "n_fc",
    "n_sl",
    "n_ch",
    "n_cu",
    "n_fs",
    "n_kn",
    "n_st",
    "n_sv",
    *[f"{stat}_{pt}" for stat in _ARSENAL_STATS for pt in _PITCH_TYPES],
    "pa",
    "bip",
    "ba",
    "est_ba",
    "est_ba_minus_ba_diff",
    "slg",
    "est_slg",
    "est_slg_minus_slg_diff",
    "woba",
    "est_woba",
    "est_woba_minus_woba_diff",
    *[f"bat_{stat}_{pt}" for stat in _ARSENAL_STATS for pt in _PITCH_TYPES],
]

### convenience alias
TRAINABLE_COLUMNS: list[str] = STATCAST_PRE_PITCH + FANGRAPH_PRE_PITCH
LABEL_COLUMN: str = "pitch_type"
