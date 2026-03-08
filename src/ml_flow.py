# ─────────────────────────────────────────────────────────────
# MLflow Setup
# ─────────────────────────────────────────────────────────────
from pathlib import Path

import dagshub
import os
import mlflow
from utils.logger import setup_logger
from config import (
    ARTIFACT_PATH,
    DAGSHUB_REPO_NAME,
    DAGSHUB_USERNAME,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
)

logger = setup_logger()


class MLflowConfig:
    """Configuration for MLflow tracking and DAGsHub integration."""

    def __init__(
        self,
    ) -> None:
        self.tracking_uri = MLFLOW_TRACKING_URI
        self.dagshub_username = DAGSHUB_USERNAME
        self.dagshub_repo_name = DAGSHUB_REPO_NAME
        self.experiment_name = MLFLOW_EXPERIMENT_NAME

    def initialize(self):
        try:
            dagshub.init(
                repo_owner=self.dagshub_username,
                repo_name=self.dagshub_repo_name,
                mlflow=True,
            )

            mlflow.set_tracking_uri(self.tracking_uri)

            logger.info(f"MLflow -> DAGsHub: {self.tracking_uri}")

        except Exception as e:
            logger.error(f"Error initializing DAGsHub: {e}")

    def setup_experiment(self):
        """Authenticate with DAGsHub and configure MLflow."""

        token = os.getenv("DAGSHUB_TOKEN")

        if token:
            self.initialize()

        else:
            logger.info("DAGSHUB_TOKEN not set – using local tracking")

        mlflow.set_experiment(self.experiment_name)

    def setup_mlflow(
        self,
        best_model_name: str,
        metrics: dict,
        roc_artifact: None,
        feature_importance_artifact: None,
    ):

        self.best_model_name = best_model_name
        self.metrics = metrics
        self.roc_artifact = roc_artifact
        self.roc_path = ARTIFACT_PATH / "roc_curve.png"
        self.feature_importance_artifact = feature_importance_artifact
        self.feature_importance_path = ARTIFACT_PATH / "feature_importance.png"

        self.initialize()

        with mlflow.start_run(run_name="customer-churn-prediction"):

            mlflow.set_tag("stage", "evaluation")
            mlflow.set_tag("best_model", self.best_model_name)

            mlflow.log_metrics(self.metrics)

            self.roc_artifact

            if self.roc_path is None:
                raise ValueError("ROC path is None!")

            mlflow.log_artifact(str(self.roc_path))

            self.feature_importance_artifact
