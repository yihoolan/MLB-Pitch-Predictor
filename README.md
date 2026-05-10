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

## Planned repo layout

Not all of these directories exist yet — they will be created as the project progresses.

```
MLB-Pitch-Predictor/
├── exploration/        # Jupyter notebooks: EDA, feature engineering, model selection
├── src/                # Modularized training & feature code (refactored out of notebooks)
├── app/                # FastAPI prediction service
├── streamlit_app/      # Streamlit UI that calls the FastAPI service
├── models/             # Serialized model artifacts (gitignored)
├── data/               # Raw / processed Statcast data (gitignored)
├── tests/              # pytest suite
├── requirements.txt        # Runtime dependencies (what ships in the container)
├── requirements-dev.txt    # Dev superset: runtime + Jupyter + plotting + lint/test/hooks
└── README.md
```

## Roadmap

1. **Exploration** — `pybaseball` data pulls, EDA, feature engineering, and model selection in Jupyter notebooks under `exploration/`.
2. **Modularization** — Lift the chosen training pipeline out of notebooks into `src/`, with proper interfaces and tests.
3. **Deployment** — Wrap the trained model in a FastAPI service, build a Streamlit frontend, containerize both with Docker, track experiments and model versions with MLflow, and wire up GitHub Actions for CI/CD. Optional AWS hosting last.
