# tests

Automated tests that run on every push and PR to `main` via GitHub Actions. All external dependencies (MLflow, pybaseball, network calls) are mocked, so the suite runs in CI without a live registry or internet access.

---

| File | Description |
| --- | --- |
| `test_api.py` | Tests API startup behavior: verifies that the `/health` endpoint reports the model as loaded and that base-runner columns are correctly encoded before being passed to the model on a `/predict` request. |
| `test_transforms.py` | Unit tests for `UsageImputer`: covers the full-row passthrough case, partial-row zero-fill for known players missing some pitch types, and global-median imputation for rookies with no prior arsenal data. |
