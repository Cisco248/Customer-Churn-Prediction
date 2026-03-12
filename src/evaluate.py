import argparse
from pathlib import Path
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
    roc_curve,
)
import numpy as np

from utils.logger import setup_logger
from config import (
    ARTIFACT_EVOLUATION_PATH,
    TRUST_LIST,
    TEST_DATA_PATH,
    TARGET_COLUMN,
    GENERATED_DATA_DIR,
    LR_EXPORT_PATH,
    RF_EXPORT_PATH,
    XGB_EXPORT_PATH,
    MLFLOW_TRACKING_URI,
    DAGSHUB_USERNAME,
    DAGSHUB_TOKEN,
    DAGSHUB_REPO_NAME,
)
from ml_flow import MLflowConfig


class FeatureEvaluationArtifacts:

    def __init__(self, loc: str | Path, name: str) -> None:
        self.location = loc
        self.model_name = name
        self.logger = setup_logger()

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
            weights = self.model.feature_importances_
            title = "Feature Importances (Tree-based)"

        elif hasattr(self.model, "coef_"):
            weights = np.abs(self.model.coef_[0])
            title = "Feature Importance (Absolute Coefficients)"

        else:
            self.logger.error(f"Model {self.model_name} has no importance attribute.")
            raise ValueError(f"Model {self.model_name} has no importance attribute.")

        importances_df = pd.DataFrame(
            {
                "feature": self.X_test.columns,
                "importance": weights,
            }
        ).sort_values(
            "importance",
            ascending=True,
        )

        importances_df.to_csv(
            GENERATED_DATA_DIR / f"{self.model_name}_data.csv",
            index=False,
        )
        mlflow.log_artifact(str(GENERATED_DATA_DIR / f"{self.model_name}_data.csv"))

        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.barh(
            importances_df["feature"],
            importances_df["importance"],
            color="skyblue",
        )
        self.ax.set_title(f"{self.model_name}: {title}")
        self.ax.set_xlabel("Importance Score")

        self.fig.savefig(
            f"{self.location}/{self.model_name}_feature.png",
            bbox_inches="tight",
        )
        plt.close(self.fig)
        mlflow.log_artifact(str(self.location))
        self.logger.info(
            f"Feature importance artifact saved for {self.model_name} ===> ✅"
        )


class RocEvaluationArtifacts:

    def __init__(self, loc: str | Path, name: str):
        self.location = loc
        self.model_name = name

    def build_evaluate_roc_curve(self, y_test=None, y_prob=None):
        self.y_axis_test = y_test
        self.y_axis_prob = y_prob

        if self.y_axis_test is None:
            raise ValueError("y_test is required to build ROC curve artifact")

        if self.y_axis_prob is None:
            raise ValueError("y_prob is required to build ROC curve artifact")

        self.fpr, self.tpr, _ = roc_curve(self.y_axis_test, self.y_axis_prob)

        self.fig, self.ax = plt.subplots()
        self.ax.plot(self.fpr, self.tpr)
        self.ax.plot([0, 1], [0, 1], "k--")

        self.ax.set_title(f"ROC Curve - {self.model_name}")
        self.ax.set_xlabel("FPR")
        self.ax.set_ylabel("TPR")

        self.path = (
            ARTIFACT_EVOLUATION_PATH / f"{self.model_name.replace(' ', '_')}_roc.png"
        )
        self.fig.savefig(self.path, bbox_inches="tight")
        plt.close(self.fig)

        mlflow.log_artifact(str(self.path))


class EvaluationModel:

    def __init__(
        self,
        model_name,
        model_path,
        ml_config: MLflowConfig,
        imp_curve_loc: str | Path,
    ) -> None:
        self.model = None
        self.metrics = {}
        self.model_name = model_name
        self.model_path = model_path
        self.ml_config = ml_config
        self.logger = setup_logger()
        self.feature_builder = FeatureEvaluationArtifacts(imp_curve_loc, model_name)
        self.roc_builder = RocEvaluationArtifacts(imp_curve_loc, model_name)

    def setup_evaluation(self) -> tuple:

        self.logger.info("🚀 ===> Evaluating Stage: Started Processing")

        self.model = sio.load(self.model_path, trusted=TRUST_LIST)

        self.logger.info(f"Evaluating: {self.model_name} ===> ℹ️")

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

        return (
            self.model,
            self.metrics,
            self.y_test,
            self.y_prob,
            self.X_test,
        )

    def run_evaluation(self):
        self.setup_evaluation()
        self.ml_config

        self.runner_name = f"Evaluation_{self.model_name}"
        with mlflow.start_run(run_name=self.runner_name):

            self.logger.info(f"Logging metrics: {self.model_name} ===> ℹ️")

            mlflow.log_metrics(self.metrics)

            self.roc_builder.build_evaluate_roc_curve(self.y_test, self.y_prob)
            self.feature_builder.build_feature_artifact(self.model, self.X_test)

            self.logger.info(f"✅ ===> Evaluating Stage: {self.model_name} Completed!")

        return (self.metrics, self.model_name)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Choose: Logistic_Regression, Random_Forest, or XG_Boost",
    )
    args = parser.parse_args()

    models = {
        "Logistic_Regression": LR_EXPORT_PATH,
        "Random_Forest": RF_EXPORT_PATH,
        "XGBoost": XGB_EXPORT_PATH,
    }

    config = MLflowConfig(
        MLFLOW_TRACKING_URI,
        DAGSHUB_TOKEN,
        DAGSHUB_USERNAME,
        DAGSHUB_REPO_NAME,
    )

    # for name, filename in models:
    #     evoluate = EvaluationModel(
    #         name,
    #         filename,
    #         config,
    #         ARTIFACT_EVOLUATION_PATH,
    #     )
    #     evoluate.run_evaluation()

    if args.model_name not in models:
        raise ValueError(
            f"Model {args.model_name} not recognized! Choose from {list(models.keys())}"
        )

    # 4. Run evaluation ONLY for the specified model
    evaluate = EvaluationModel(
        args.model_name,
        models[args.model_name],
        config,
        ARTIFACT_EVOLUATION_PATH,
    )
    evaluate.run_evaluation()
