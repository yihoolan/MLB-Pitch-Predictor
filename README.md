# MLB Pitch Predictor

Predicts the **type of the next pitch** (fastball, slider, curveball, ...) given the current game situation and pitcher/batter matchup, using Statcast data from [`pybaseball`](https://github.com/jldbc/pybaseball). The model is a LightGBM classifier trained on pitch-by-pitch Statcast data, served through a FastAPI backend and a Streamlit dashboard.

**Stack:** `pybaseball` · `lightgbm` · `scikit-learn` · `optuna` · `mlflow` · `fastapi` · `uvicorn` · `streamlit` · Docker · GitHub Actions

---

## Running the dashboard (no setup required)

The easiest way to run the dashboard is with Docker. 

```bash
git clone https://github.com/yihoolan/MLB-Pitch-Predictor.git
cd MLB-Pitch-Predictor
docker compose pull
docker compose up
```

Open `http://localhost:8501` in your browser. The FastAPI backend starts alongside the Streamlit app automatically.

> **Note:** On first startup the API downloads current-season arsenal stats from Baseball Reference. This takes a minute but only happens once per container session.

---

## Developer setup

Requires Python 3.11.

```bash
python3.11 -m venv .venv
source .venv/bin/activate            # PowerShell: .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements-dev.txt  # model-development + notebooks + lint/test tools
```

### Running locally without Docker

Start the API and the dashboard in separate terminals from the project root:

```bash
uvicorn app.main:app --reload --port 8001
streamlit run streamlit_app/app.py
```

### Testing

```bash
pytest tests/ -q
```

CI runs the same command on every push and PR to `main` via `.github/workflows/ci.yml`.

---

## Training

Training pulls Statcast data, tunes a LightGBM classifier with Optuna, and registers the best model in the local MLflow registry under `PitchTypeClassifier`.

```bash
# Full retrain from scratch
python -m training.train --mode full

# Incremental update on recent seasons only
python -m training.train --mode incremental
```

Hyperparameter tuning runs separately:

```bash
python -m training.tune
```

After training, promote the best run to Production in the MLflow UI (`mlflow ui`) or via the promotion script in `scripts/`.

### Updating the Docker images after a retrain

Once a new model is promoted to Production, rebuild and push the images to GHCR:

```bash
bash scripts/rebuild_docker.sh
```

This rebakes `mlruns/` into the API image and pushes both images. Users get the updated model on their next `docker compose pull && docker compose up`.

> **Note:** Incremental training loads the base model from the local `mlruns/` directory, so it must run on the same machine that holds `mlruns/`. If you're switching machines, copy `mlruns/` over first. Full retrains (`--mode full`) have no such dependency — `mlruns/` is created from scratch.

Requires a one-time login before the first push:

```bash
docker login ghcr.io -u yihoolan --password <PAT with write:packages scope>
```

---

## Repo layout

```
MLB-Pitch-Predictor/
├── exploration/        # Jupyter notebooks: EDA, feature engineering, model selection
├── training/           # LightGBM training, tuning, evaluation, and MLflow registration
├── app/                # FastAPI prediction service
│   └── routers/        # Route handlers for /players and /predict
├── streamlit_app/      # Streamlit dashboard — calls the FastAPI service over HTTP
├── utils/              # Shared feature definitions and preprocessing transforms
├── scripts/            # Utility scripts (production model selection, Docker rebuild)
├── data/               # Raw / processed Statcast data (gitignored)
├── mlruns/             # MLflow experiment tracking and model registry (gitignored)
├── Dockerfile          # API service image (mlruns/ baked in at build time)
├── Dockerfile.streamlit
├── docker-compose.yml
├── requirements.txt        # Runtime dependencies
└── requirements-dev.txt    # Dev superset: runtime + Jupyter + lint/test tools
```

---

## Architecture & Code Flow

**Offline training pipeline.** `training/data.py` pulls Statcast seasons via pybaseball and merges prior-year pitcher and batter arsenal stats (via `utils/enrichment.py`), producing temporal train/val/test splits. `training/train.py` fits a LightGBM classifier and wraps it in a `PitchPredictor` pyfunc (`training/predictor.py`) before logging everything to MLflow.

**Hyperparameter tuning and promotion.** `training/tune.py` drives an Optuna search across `OPTUNA_N_TRIALS` trials, each logged as a nested MLflow run. After any training run, `training/promote.py` compares the challenger's log-loss against the current Production version and transitions stages if the new model wins.

**API startup and request path.** On startup, `app/model.py` loads the Production model from MLflow. Prediction requests are enriched with live arsenal stats by `app/enrichment.py`, then passed to `routers/predict.py`, which returns pitch-type probabilities.

**Streamlit → API.** A front-end dashboard powered by `streamlit_app/app.py` that routes requests to the FastAPI service to display real-time pitch probabilities.

### Also in this repo

**Exploration** — Jupyter notebooks in `exploration/` cover the EDA and feature-selection arc that determined which Statcast columns to use in production. Findings are summarized in `00_exploration_findings.md`.

**Tests** — `test_api.py` and `test_transforms.py` provide simple tests for API startup and data transformation functions.

**Docker** — `Dockerfile` and `Dockerfile.streamlit` package the API and dashboard with all dependencies baked in. `docker-compose.yml` wires the two services together for instant front-end replication with a single `docker compose up`.

## Recognition
LLM is prompted sporadically for clear documentation and boilerplate functionalities