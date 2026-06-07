# Exploration Findings Summary

This document synthesizes the June 2024 exploration work on Statcast pitch data enriched with 2023 prior-year arsenal stats. The sample covers roughly 114,000 labeled pitches from one month; findings are specific to that window unless validated on a broader pull.

---

## Findings — data quality

There are very few usable data for rare pitch types (knuckleball). This suggests the model might not do so well on a knuckleball pitcher. In the June 2024 slice, only 270 pitches (0.24%) were classified as KN; prior-year enrichment columns for knuckleballs (`pitch_usage_KN`, `pitches_KN`, and related outcome stats) are roughly 99.5% null. Sweeper (SV) is similarly sparse at 563 pitches and ~98% null on its enrichment columns.

Quite a few features have non-trivial missingness. More than 70% of trainable features have more than 20% missingness — 276 of 388 enriched columns, and roughly three-quarters of the 281 available EDA columns by tier composition. This comes from multiple sources: rookie players, players with not enough appearance, and unavailable Statcast measures. The enrichment pipeline joins 2023 arsenal tables with a minimum-PA threshold of 25, so call-ups and low-volume players often have no prior-year row; pitcher-side `pitch_usage_FF` coverage is 68.6% of pitches. Fortunately, the most important features tend to persist quite well.

---

## Findings — pitch selection

Fastball (FF) is the dominant pitch type followed by a suite of off-speeds. FF accounts for 32.5% of pitches in the sample, then SI (16.3%), SL (14.9%), and CH (10.0%); everything else is single-digit share.

The 2×2 matrix of pitcher/batter handedness seem quite useful for identifying enhanced probabilities for certain off-speeds. For instance, it appears that slider usage is much increased when pitcher and batter has same handedness, and similarly with changeup for different handedness. Fastball also seems more likely for different handedness, though not to the effect of changeup.

Pitch count is a signal for pitch type. Fastball/sinker is more likely in hitter's count and off-speeds are more likely in pitcher's count. Balls and strikes rank among the highest game-state importances in the random forest experiments.

Fastball/sinker usage decreases as batter order increases (less fastball is used as pitcher faces the same batter more times in a game).

It appears that larger run-differential does decrease fastball usage.

Prior pitch arsenal distribution is a very strong predictor of pitch usage. This will serve as a very informative prior in inference. Per-pitcher scatter plots of prior-year `pitch_usage_*` against actual June 2024 mix show strong linear fit for common pitch types; Set B feature importances are dominated by pitcher and batter usage columns.

---

## Findings — feature importance

While there are a lot of batter/pitcher outcomes (~250 of them), they are not particularly informative for pitch type. PCA also does not help that much here, as the explained variance of top principal components is quite low (and does not improve performance). The outcome blocks total 252 columns (140 pitcher, 112 batter); five principal components explain only about 44% cumulative variance for each block.

Adding the full outcome blocks does not beat a leaner feature set. Sets C, D, and E reach OOB accuracy of roughly 0.37–0.38 despite 169–281 features, at or below Set B at ~0.39 — outcome statistics add complexity without improving out-of-bag performance.

A generic model trained only on 29 features (candidate B) performs similarly compared to using the full trainable features in terms of OOB accuracy. Set A (game state only, 11 features) sits around 0.18 OOB; Set B jumps to ~0.39; Sets C/D/E with all outcomes plateau slightly below that.

In light of the above, we can just do model training on candidate B — 11 game-state columns plus 10 pitcher usage and 10 batter usage columns, encoding to 29 random forest features.

---

## The 10-class problem

The label space covers ten Statcast pitch types: FF, SI, FC, SL, CH, CU, FS, KN, ST, and SV. This is not a balanced classification task. FF runs 32.5% of the sample; KN is 0.24% and SV about 0.5%.

Useful baselines for any model evaluation: random guessing is ~10% accuracy; always predicting FF is ~33%; the exploratory Set B random forest with balanced class weights reaches ~39% OOB. The model clearly beats naive baselines, but aggregate accuracy hides pain on tail classes — KN and SV are under-represented in both the label distribution and enrichment coverage, which compounds prediction difficulty for knuckleball and sweeper specialists.

---

## Game state hierarchy

Random forest experiments on candidate feature sets reveal a clear tier structure.

Game state alone (Set A, 11 columns) reaches ~0.18 OOB — count (`balls`, `strikes`), sequencing (`pitch_number`), score context (`bat_score_diff`), handedness (`stand`, `p_throws`), inning, outs, and base runners carry signal but are not enough on their own.

Adding prior-year usage priors (Set B, 29 columns) lifts OOB to ~0.39. The `pitch_usage_*` and `bat_pitch_usage_*` columns provide the largest gain; arsenal distribution is the dominant prior for pitch selection.

Stacking on the ~250 outcome statistics (Sets C, D, E, 141–281 columns) does not improve on Set B — OOB sits around 0.37–0.38, slightly below the lean model.

Within game state, mean decrease in impurity ranks `balls`, `strikes`, `pitch_number`, and `bat_score_diff` highest; `inning` and binarized base-state columns contribute less. Handedness encoding is validated by the pitch distribution EDA on the 2×2 matchup matrix.

---

## Missingness mechanics

High null rates in the raw audit do not always mean unusable features. Three distinct mechanisms drive missingness.

Enrichment join gaps: prior-year arsenal tables are left-joined on pitcher and batter IDs with `min_pa=25`. Rookies, call-ups, and low-appearance players lack rows — roughly 31% of pitches have no pitcher enrichment. Batter-side tier median null is ~32% versus ~74% for pitcher usage and outcome tiers.

Rare-pitch structural gaps: almost no player has prior-year KN or SV stats, so those columns reach ~99% null. Thirty EDA columns (mostly batter KN/SV) are absent from the enriched frame entirely.

Structural Statcast nulls: `on_1b`, `on_2b`, and `on_3b` show ~91% null in raw data because empty bases are stored as NaN, not zero. The pipeline binarizes these to 0/1 before modelling — the high null rate is encoding convention, not lost data. Deprecated legacy fields (`spin_dir`, etc.) are fully null and sit outside trainable columns.

Before modelling, usage columns pass through a zero-vs-global-prior imputer: zero when a player has any usage row but a specific pitch type is missing; stratified median prior when the player has no enrichment row at all. Numeric game state uses median imputation. This handling is why the most important features persist quite well despite tier-level missingness in the raw audit.

---

## Modelling recommendation

Train on candidate B — game state plus pitcher and batter usage priors, 29 encoded features. Skip the ~250 outcome columns and PCA compression; they add noise without OOB gain.

Watch rare pitch types closely. KN and SV lack both label volume and enrichment coverage; expect weak predictions for knuckleball and sweeper specialists unless class handling (weighting, grouping, or exclusion) is addressed explicitly. Prior-year usage is a strong prior and many in game-state features can help enhance that prior.
