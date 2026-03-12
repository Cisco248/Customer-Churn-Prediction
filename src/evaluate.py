from src.utils.logger import setup_logger
import skops.io as sio
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    roc_curve,
)
from src.config import *
from src.ml_flow import MLflowConfig


class FeatureEvaluationArtifacts:

    def __init__(self, loc: str | Path) -> None:
        self.location = loc

    def build_feature_artifact(self, model=None, X_test=None):
        self.model = model
        self.X_test = X_test

        if self.X_test is None:
            raise ValueError("X_test is required to build feature importance artifact")

        if self.model is None:
            raise ValueError("Model is required to build feature importance artifact")

        if self.location is None:
            raise ValueError("Location is required to save feature importance artifact")

        if hasattr(self.model, "feature_importances_"):

            self.importance = pd.DataFrame(
                {
                    "feature": self.X_test.columns,
                    "importance": self.model.feature_importances_,
                }
            ).sort_values("importance", ascending=False)

            self.fig, self.ax = plt.subplots()
            self.ax.barh(self.importance["feature"], self.importance["importance"])
            self.ax.set_title("Feature Importances")

            self.fig.savefig(self.location)
            plt.close(self.fig)

            mlflow.log_artifact(str(self.location))


class RocEvaluationArtifacts:

    def __init__(self, loc=None):
        self.location = loc

    def build_roc_artifact(self, y_test=None, y_prob=None):
        self.y_axis_test = y_test
        self.y_axis_prob = y_prob

        if self.y_axis_test is None:
            raise ValueError("y_test is required to build ROC curve artifact")

        if self.y_axis_prob is None:
            raise ValueError("y_prob is required to build ROC curve artifact")

        if self.location is None:
            raise ValueError("ROC location is required to save ROC curve artifact")

        self.fpr, self.tpr, _ = roc_curve(self.y_axis_test, self.y_axis_prob)

        self.fig, self.ax = plt.subplots()
        self.ax.plot(self.fpr, self.tpr)
        self.ax.plot([0, 1], [0, 1], "k--")

        self.ax.set_title("ROC Curve")
        self.ax.set_xlabel("FPR")
        self.ax.set_ylabel("TPR")

        self.fig.savefig(self.location)
        plt.close(self.fig)

        mlflow.log_artifact(self.location)


class EvaluationModel:

    def __init__(
        self,
        model_name,
        model_path,
        ml_config: MLflowConfig,
    ) -> None:
        self.model_name = model_name
        self.model_path = model_path
        self.ml_config = ml_config
        self.logger = setup_logger()
        self.feature_builder = FeatureEvaluationArtifacts(
            ARTIFACT_PATH / "feature_importance.png"
        )
        self.roc_builder = RocEvaluationArtifacts(ARTIFACT_PATH / "roc_curve.png")

    def setup_evaluation(self) -> tuple:

        self.model = sio.load(self.model_path, trusted=TRUST_LIST)

        self.logger.info(f"\n{'='*50}\n{self.model_name}\n{'='*50}")

        self.test_df = pd.read_csv(TEST_DATA_PATH)

        self.X_test = self.test_df.drop(columns=[TARGET_COLUMN])
        self.y_test = self.test_df[TARGET_COLUMN]

        self.y_pred = self.model.predict(self.X_test)
        self.y_prob = self.model.predict_proba(self.X_test)[:, 1]

        self.metrics = {
            "accuracy": round(float(accuracy_score(self.y_test, self.y_pred)), 4),
            "precision": round(float(precision_score(self.y_test, self.y_pred)), 4),
            "recall": round(float(recall_score(self.y_test, self.y_pred)), 4),
            "f1": round(float(f1_score(self.y_test, self.y_pred)), 4),
            "roc_auc": round(float(roc_auc_score(self.y_test, self.y_prob)), 4),
        }

        self.logger.info("\nEvaluation Metrics:")

        for k, v in self.metrics.items():
            self.logger.info(f"{k}: {v}")

        report = classification_report(
            self.y_test, self.y_pred, target_names=TARGET_NAMES
        )

        self.logger.info(f"{REPORT_TITLE}\n{report}")

        return (
            self.model,
            self.metrics,
            self.y_test,
            self.y_prob,
            self.X_test,
        )

    def run_evaluation(self):
        self.ml_config

        self.logger.info("Running Final Evaluation on All Models")

        self.results = []

        self.setup_evaluation()

        self.results.append(
            {
                "name": self.model_name,
                "Metrics": self.metrics,
                "Y-Test": self.y_test,
                "Y-Prob": self.y_prob,
                "X-Test": self.X_test,
            }
        )

        self.best = max(self.results, key=lambda x: x["Metrics"]["roc_auc"])
        self.best_model = self.best["model"]
        self.best_name = self.best["name"]
        self.best_metrics = self.best["metrics"]

        self.logger.info(
            f"\nBest Model: {self.best_name} | ROC-AUC: {self.best_metrics['roc_auc']}"
        )

        with mlflow.start_run(run_name=f"[{self.best_name} Final Evaluation]"):
            mlflow.log_metrics(self.best_metrics)
            self.roc_builder.build_roc_artifact(self.y_test, self.y_prob)
            self.feature_builder.build_feature_artifact(self.model_path, self.X_test)

        return (
            self.best_model,
            self.best_metrics,
            self.best_name,
        )


if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Add project root to path for proper imports
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    logistic_evo = EvaluationModel(
        "Logistic_Regression",
        MODELS_DIR / "logistic_regression.skops",
        MLflowConfig(
            MLFLOW_TRACKING_URI, DAGSHUB_TOKEN, DAGSHUB_USERNAME, DAGSHUB_REPO_NAME
        ),
    )
    logistic_evo.setup_evaluation()
    logistic_evo.run_evaluation()

    _random_evo = EvaluationModel(
        "Random_Forest",
        MODELS_DIR / "random_forest.skops",
        MLflowConfig(
            MLFLOW_TRACKING_URI, DAGSHUB_TOKEN, DAGSHUB_USERNAME, DAGSHUB_REPO_NAME
        ),
    )
    _random_evo.setup_evaluation()
    _random_evo.run_evaluation()

    _xgboost_evo = EvaluationModel(
        "XG_Boost",
        MODELS_DIR / "xgboost.skops",
        MLflowConfig(
            MLFLOW_TRACKING_URI, DAGSHUB_TOKEN, DAGSHUB_USERNAME, DAGSHUB_REPO_NAME
        ),
    )
    _xgboost_evo.setup_evaluation()
    _xgboost_evo.run_evaluation()

    # with mlflow.start_run(run_name="customer-churn-prediction"):

    #     mlflow.set_tag("stage", "evaluation")
    #     mlflow.set_tag("best_model", self.best_model_name)

    #     mlflow.log_metrics(self.metrics)

    # self.roc_artifact

    # if self.roc_path is None:
    #     raise ValueError("ROC path is None!")

    # mlflow.log_artifact(str(self.roc_path))

    # self.feature_importance_artifact

    # self.artifacts_builder.build_roc_curve(
    #     self.best["y_test"],
    #     self.best["y_prob"],
    #     loc=ARTIFACT_PATH / "roc_curve.png",
    # )

    # self.artifacts_builder.build_feature_importance_artifact(
    #     self.best_model,
    #     self.best["X_test"],
    #     loc=ARTIFACT_PATH / "feature_importance.png",
    # )

    # def __init__(
    #     self, uri: str, token: str | None, username: str | None, repo_name: str | None
    # ) -> None:
    #     self.tracking_uri = uri
    #     self.token = token
    #     self.dagshub_username = username
    #     self.dagshub_repo_name = repo_name

    #     try:
    #         if not self.token:
    #             logger.info("DAGsHub authentication unsuccessful")

    #         logger.info("DAGsHub authentication successful")
    #         dagshub.init(
    #             repo_owner=self.dagshub_username,
    #             repo_name=self.dagshub_repo_name,
    #             mlflow=True,
    #         )
    #         mlflow.set_tracking_uri(self.tracking_uri)
    #         logger.info(f"MLflow: {self.tracking_uri}")

    #     except Exception as e:
    #         logger.error(f"Error initializing DAGsHub: {e}")
