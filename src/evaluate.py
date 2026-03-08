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
from xgboost import Booster, XGBClassifier
from config import *
from utils.logger import setup_logger
from ml_flow import MLflowConfig

ml_config = MLflowConfig()
logger = setup_logger()


class BuildEvolutionArtifacts:

    def __init__(self) -> None:
        pass

    def build_feature_importance_artifact(
        self,
        model: pd.DataFrame,
        X_test: pd.DataFrame,
        loc,
    ):
        self.X_test = X_test
        self.model = model
        self.loc = loc

        if hasattr(model, "feature_importances_"):

            self.importance = (
                pd.DataFrame(
                    {
                        "feature": self.X_test.columns,
                        "importance": self.model.feature_importances_,
                    }
                )
                .sort_values("importance", ascending=False)
                .head(20)
            )

            fig, ax = plt.subplots()
            ax.barh(self.importance["feature"], self.importance["importance"])
            ax.set_title("Top 20 Feature Importances")
            fi_path = self.loc
            fig.savefig(fi_path)
            plt.close(fig)
            mlflow.log_artifact(str(fi_path))

    def build_roc_curve(self, y_test, y_prob, loc) -> None:
        self.y_test = y_test
        self.y_prob = y_prob
        self.loc = loc

        fpr, tpr, _ = roc_curve(self.y_test, self.y_prob)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr)
        ax.plot([0, 1], [0, 1], "k--")
        ax.set_title("ROC Curve")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        roc_path = self.loc
        fig.savefig(roc_path)
        plt.close(fig)


def evaluate_model(model_name: str, model_path):

    logger.info(f"\n{'='*50}\n{model_name}\n{'='*50}")

    model = sio.load(
        model_path,
        trusted=TRUST_LIST,
    )

    test_df = pd.read_csv(TEST_DATA_PATH)
    X_test = test_df.drop(columns=[TARGET_COLUMN]).values
    y_test = test_df[TARGET_COLUMN]

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(float(precision_score(y_test, y_pred)), 4),
        "recall": round(float(recall_score(y_test, y_pred)), 4),
        "f1": round(float(f1_score(y_test, y_pred)), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
    }

    logger.info("\nEvaluation Metrics:")
    for k, v in metrics.items():
        logger.info(f"{k}: {v}")

    report_text = classification_report(y_test, y_pred, target_names=TARGET_NAMES)

    logger.info(f"{REPORT_TITLE}\n{report_text}")

    return model, metrics, y_test, y_prob, X_test


def run_evaluation():

    ml_config.setup_experiment

    logger.info("Running Final Evaluation on All Models")

    for name, path in EVALUTION_MODELS.items():

        model, metrics, y_test, y_prob, X_test = evaluate_model(name, path)

        if metrics["roc_auc"] > BEST_AUC:
            best_auc = metrics["roc_auc"]
            best_model_name = name
            best_results = (model, metrics, y_test, y_prob, X_test)

    logger.info(f"\nBest Model: {best_model_name} | ROC-AUC: {best_auc}")

    if best_results is None:
        raise ValueError("Best results not found!")

    model, metrics, y_test, y_prob, X_test = best_results

    if best_model_name is None:
        raise ValueError("Best model name is None!")

    artifacts_builder = BuildEvolutionArtifacts()

    ml_config.setup_mlflow(
        best_model_name=best_model_name,
        metrics=metrics,
        roc_artifact=artifacts_builder.build_roc_curve(
            y_test,
            y_prob,
            loc=ARTIFACT_PATH / "roc_curve.png",
        ),
        feature_importance_artifact=artifacts_builder.build_feature_importance_artifact(
            model,
            X_test,
            loc=ARTIFACT_PATH / "feature_importance.png",
        ),
    )

    logger.info("Evaluation Complete")


if __name__ == "__main__":
    run_evaluation()
