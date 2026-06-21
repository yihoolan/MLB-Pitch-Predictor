# scripts

Utility scripts invoked by CI/CD workflows rather than by the application itself.

---

| File | Description |
| --- | --- |
| `get_production_version.py` | Queries the local MLflow registry for the current Production model version and writes `prod_version` and `next_year` to `$GITHUB_OUTPUT` for use in the monthly incremental training GitHub Actions workflow. |
| `rebuild_docker.sh` | Builds and pushes the API and Streamlit images to GHCR after a retrain, rebaking `mlruns/` into the API image so users get the updated model on their next `docker compose pull`. |
