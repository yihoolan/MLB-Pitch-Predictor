# exploration

Jupyter notebooks covering the EDA and feature-selection arc that determined `MODEL_FEATURES` for the production classifier.

---

| File | Description |
| --- | --- |
| `00_exploration_findings.md` | Written summary of key findings from the EDA arc: data quality issues, pitch-type class imbalance, and the feature-set decision that led to choosing Set B (`MODEL_FEATURES`). |
| `01_pybaseball_installation_check.ipynb` | Confirms that `pybaseball` can pull Statcast and Fangraph data end-to-end in this environment, with caching enabled. |
| `02_data_quality.ipynb` | Audits the raw Statcast frame for missingness and column classification across all 311 EDA columns, organized into deprecated, post-pitch, logistics, and pre-pitch tiers. |
| `03_pitch_distribution_eda.ipynb` | Visual exploration of how pitch selection varies by count, handedness matchup, score differential, inning, and prior-year arsenal usage rates. |
| `04_feature_importance_ml.ipynb` | Compares candidate feature sets A–E using Mutual Information and Random Forest importance to justify the selection of Set B as `MODEL_FEATURES`. |
