# utils

Shared preprocessing and feature-definition code imported by both `training/` and `app/`.

---

| File | Description |
| --- | --- |
| `feature_names.py` | Single source of truth for all column name lists (Statcast categories, pitch types, arsenal stat families, candidate feature sets A–E) and the production `MODEL_FEATURES` constant. |
| `transforms.py` | Stateless and stateful preprocessing: `binarize_bases` converts runner columns to 0/1, `UsageImputer` handles rookie zero-vs-median imputation, and `build_feature_matrix` assembles a numeric array for notebook experiments. |
| `enrichment.py` | Pulls prior-year pitcher and batter arsenal stats from pybaseball, pivots them wide, and merges onto a raw Statcast DataFrame for offline training data preparation. This is the offline counterpart to `app/enrichment.py`, which performs the same stat fetch live at inference time. |
