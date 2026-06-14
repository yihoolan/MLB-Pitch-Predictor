"""Auto-promotion logic for the MLflow model registry.

promote_if_better compares the challenger's log_loss against the current
Production model and transitions stages if the challenger wins.  Both
training/train.py and training/tune.py call this after registering a new version.
"""

from __future__ import annotations

from mlflow.tracking import MlflowClient

from training.config import REGISTERED_MODEL_NAME


def promote_if_better(new_version: str, new_log_loss: float) -> bool:
    """Promote new_version to Production if its log_loss beats the current champion.

    - If no Production model exists yet, promotes unconditionally.
    - If a Production model exists and new_log_loss is lower, archives the old
      version and promotes the new one.
    - Returns True if a promotion occurred, False otherwise.
    """
    client = MlflowClient()
    prod = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=["Production"])

    if not prod:
        client.transition_model_version_stage(REGISTERED_MODEL_NAME, new_version, "Production")
        print(f"  No prior Production model — promoted v{new_version} directly.")
        return True

    prod_version = prod[0]
    prod_log_loss = float(client.get_run(prod_version.run_id).data.metrics["log_loss"])

    if new_log_loss < prod_log_loss:
        client.transition_model_version_stage(REGISTERED_MODEL_NAME, prod_version.version, "Archived")
        client.transition_model_version_stage(REGISTERED_MODEL_NAME, new_version, "Production")
        print(
            f"  Promoted v{new_version} to Production "
            f"(log_loss {new_log_loss:.4f} < {prod_log_loss:.4f}). "
            f"Archived v{prod_version.version}."
        )
        return True

    print(f"  v{new_version} NOT promoted " f"(log_loss {new_log_loss:.4f} >= current {prod_log_loss:.4f}).")
    return False
