"""Query the local MLflow registry and write Production model info to GITHUB_OUTPUT.

Called by .github/workflows/monthly_train.yml to resolve the current Production
version and the target data year before launching incremental training.

Outputs (written to $GITHUB_OUTPUT):
    prod_version  — MLflow registry version string of the current Production model
    next_year     — calendar year of data to train on (current year)
"""

from __future__ import annotations

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import mlflow
from mlflow.tracking import MlflowClient

from settings import settings

mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
client = MlflowClient()
prod = client.get_latest_versions(settings.registered_model_name, stages=["Production"])

if not prod:
    print(f"ERROR: no Production model found for '{settings.registered_model_name}'", file=sys.stderr)
    sys.exit(1)

prod_version = prod[0].version
next_year = str(datetime.now().year)

output_file = os.environ.get("GITHUB_OUTPUT", "")
if output_file:
    with open(output_file, "a") as f:
        f.write(f"prod_version={prod_version}\n")
        f.write(f"next_year={next_year}\n")
else:
    # Fallback for local testing
    print(f"prod_version={prod_version}")
    print(f"next_year={next_year}")
