# scripts

Utility scripts invoked manually or by CI/CD workflows rather than by the application itself.

---

| File | Description |
| --- | --- |
| `export_model.py` | Exports the current Production model artifacts from the local MLflow registry to `model/`. Run after promoting a new model, then commit and push `model/` to trigger a Docker image rebuild. |
| `get_production_version.py` | Queries the local MLflow registry for the current Production model version and writes `prod_version` and `next_year` to `$GITHUB_OUTPUT` for use in the monthly incremental training GitHub Actions workflow. |
