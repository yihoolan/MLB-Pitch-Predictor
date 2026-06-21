"""Export the Production model artifacts from the local MLflow registry to model/.

Run this after promoting a new model to Production, then commit the result:

    python scripts/export_model.py
    git add model/
    git commit -m "chore(model): export Production model vN"
    git push

The Docker image baked by CI will contain these files and the API loads them
directly at startup — no MLflow registry access required at serve time.
"""

from __future__ import annotations

import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import mlflow  # noqa: E402 — path must be set first
from mlflow.tracking import MlflowClient

from settings import settings

mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
client = MlflowClient()

prod = client.get_latest_versions(settings.registered_model_name, stages=["Production"])
if not prod:
    print(f"ERROR: no Production model found for '{settings.registered_model_name}'", file=sys.stderr)
    sys.exit(1)

version = prod[0].version
run_id = prod[0].run_id
print(f"Exporting Production model v{version} (run {run_id})...")

dest = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
os.makedirs(dest, exist_ok=True)

tmp = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="model/artifacts")

shutil.copy(os.path.join(tmp, "model.lgb"), os.path.join(dest, "model.lgb"))
shutil.copy(os.path.join(tmp, "preprocessor.pkl"), os.path.join(dest, "preprocessor.pkl"))

with open(os.path.join(dest, "version.txt"), "w") as f:
    f.write(f"{version}\n")

print(f"  model/model.lgb")
print(f"  model/preprocessor.pkl")
print(f"  model/version.txt  (v{version})")
print("Done. Commit model/ and push to trigger a Docker image rebuild.")
