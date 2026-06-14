# app

FastAPI service that loads the Production model from the MLflow registry at startup and exposes pitch prediction and player search over HTTP.

---

| File | Description |
| --- | --- |
| `main.py` | FastAPI application entry point; loads the Production model and arsenal cache at startup and mounts the `/health` and `/reload` endpoints. |
| `model.py` | Defines `ModelRegistry`, a thin wrapper that pulls the Production model from the local MLflow registry at startup and supports hot-reload without restarting the server. |
| `enrichment.py` | Player name search against the Chadwick register and per-season arsenal stat fetching; both are cached in-process so repeated calls within a server session are free. |
| `schemas.py` | Pydantic models for request and response validation: `GameStateRequest`, `PitchProbabilities`, `PlayerMatch`, and `HealthResponse`. |
| `routers/players.py` | `GET /players` — fuzzy player name search filtered by role (pitcher or batter), returning up to 10 `PlayerMatch` results. |
| `routers/predict.py` | `POST /predict` — enriches a game-state request with prior-year arsenal stats and runs the Production model to return pitch-type probabilities. |
