# MLB-Pitch-Predictor

A personal end-to-end ML project that predicts the **type of the next pitch** (fastball, slider, curveball, ...) given game context, using Statcast data fetched via [`pybaseball`](https://github.com/jldbc/pybaseball). This is a reproducible end-to-end pipeline with the following structure: EDA in notebooks, modular training code, model versioning with MLflow, a FastAPI prediction service, a Streamlit UI, all containerized with Docker and wired up to GitHub for CI/CD.

## Stack

- **Data**: `pybaseball` (Statcast pulls)
- **Modeling**: `scikit-learn`, `pandas`, `numpy`
- **Experiment tracking & registry**: `mlflow`
- **API**: `fastapi` + `uvicorn`
- **UI**: `streamlit`
- **Containerization**: Docker (added in a later step)
- **CI/CD**: GitHub Actions (added in a later step)
- **Hosting (optional)**: AWS

## Reproducible setup

### Local development

Requires Python 3.11.

```bash
python3.11 -m venv .venv
source .venv/bin/activate            # PowerShell: .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements-dev.txt   # runtime + notebooks + tooling
```

### Runtime-only (mirrors what Docker will install)

```bash
pip install -r requirements.txt
```

### Docker

Coming in the deployment step. Will follow the pattern:

```bash
docker build -t mlb-pitch-predictor .
docker compose up
```

## Repo layout

```
MLB-Pitch-Predictor/
├── exploration/        # Jupyter notebooks: EDA, feature engineering, model selection
├── training/           # LightGBM training, tuning, evaluation, and MLflow registration
├── app/                # FastAPI prediction service
│   └── routers/        # Route handlers for /players and /predict
├── streamlit_app/      # Streamlit UI — calls the FastAPI service over HTTP
├── utils/              # Shared feature definitions and preprocessing transforms
├── scripts/            # Utility scripts for CI/CD workflows
├── data/               # Raw / processed Statcast data (gitignored)
├── mlruns/             # MLflow experiment tracking and model registry (gitignored)
├── requirements.txt        # Runtime dependencies (what ships in the container)
├── requirements-dev.txt    # Dev superset: runtime + Jupyter + plotting + lint/test/hooks
└── README.md
```
