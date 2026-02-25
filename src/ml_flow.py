# ─────────────────────────────────────────────────────────────
# MLflow Setup
# ─────────────────────────────────────────────────────────────

import os
import mlflow
from utils.logger import setup_logger
from config import (
    DAGSHUB_REPO_NAME,
    DAGSHUB_USERNAME,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
)

logger = setup_logger()


def setup_mlflow():
    """Authenticate with DAGsHub and configure MLflow."""

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    token = os.getenv("DAGSHUB_TOKEN")

    if token:
        import dagshub

        dagshub.init(
            repo_owner=DAGSHUB_USERNAME,
            repo_name=DAGSHUB_REPO_NAME,
            mlflow=True,
        )
        logger.info(f"MLflow -> DAGsHub: {MLFLOW_TRACKING_URI}")
    else:
        logger.info("DAGSHUB_TOKEN not set – using local tracking")

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
