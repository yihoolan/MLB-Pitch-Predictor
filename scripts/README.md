# scripts

Utility scripts invoked by CI/CD workflows rather than by the application itself.

---

| File | Description |
| --- | --- |
| `get_production_version.py` | Queries the local MLflow registry for the current Production model version and writes `prod_version` and `next_year` to `$GITHUB_OUTPUT` for use in the monthly incremental training GitHub Actions workflow. |
