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
from config import *
from utils.logger import setup_logger
from ml_flow import MLflowConfig

ml_config = MLflowConfig()
logger = setup_logger()


class BuildEvaluationArtifacts:

    def build_feature_importance_artifact(self, model, X_test, loc):

        if hasattr(model, "feature_importances_"):

            importance = pd.DataFrame(
                {
                    "feature": X_test.columns,
                    "importance": model.feature_importances_,
                }
            ).sort_values("importance", ascending=False)

            fig, ax = plt.subplots()
            ax.barh(importance["feature"], importance["importance"])
            ax.set_title("Feature Importances")

            fig.savefig(loc)
            plt.close(fig)

            mlflow.log_artifact(str(loc))

    def build_roc_curve(self, y_test, y_prob, loc):

        fpr, tpr, _ = roc_curve(y_test, y_prob)

        fig, ax = plt.subplots()
        ax.plot(fpr, tpr)
        ax.plot([0, 1], [0, 1], "k--")

        ax.set_title("ROC Curve")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")

        fig.savefig(loc)
        plt.close(fig)

        mlflow.log_artifact(str(loc))


def evaluate_model(model_name, model_path):

    logger.info(f"\n{'='*50}\n{model_name}\n{'='*50}")

    model = sio.load(
        model_path,
        trusted=TRUST_LIST,
    )

    test_df = pd.read_csv(TEST_DATA_PATH)

    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred)), 4),
        "recall": round(float(recall_score(y_test, y_pred)), 4),
        "f1": round(float(f1_score(y_test, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
    }

    logger.info("\nEvaluation Metrics:")
    for k, v in metrics.items():
        logger.info(f"{k}: {v}")

    report = classification_report(y_test, y_pred, target_names=TARGET_NAMES)
    logger.info(f"{REPORT_TITLE}\n{report}")

    return model, metrics, y_test, y_prob, X_test


def run_evaluation():

    ml_config.setup_experiment()

    logger.info("Running Final Evaluation on All Models")

    results = []

    for name, path in EVALUTION_MODELS.items():

        model, metrics, y_test, y_prob, X_test = evaluate_model(name, path)

        results.append(
            {
                "name": name,
                "model": model,
                "metrics": metrics,
                "y_test": y_test,
                "y_prob": y_prob,
                "X_test": X_test,
            }
        )

    best = max(results, key=lambda x: x["metrics"]["roc_auc"])

    best_model = best["model"]
    best_name = best["name"]
    best_metrics = best["metrics"]

    print(f"Best Model: {best_name}")
    print(f"ROC-AUC: {best_metrics['roc_auc']}")

    logger.info(f"\nBest Model: {best_name} | ROC-AUC: {best_metrics['roc_auc']}")

    artifacts_builder = BuildEvaluationArtifacts()

    with mlflow.start_run(run_name=f"[{best_name} Final Evaluation]"):

        mlflow.log_metrics(best_metrics)

        artifacts_builder.build_roc_curve(
            best["y_test"],
            best["y_prob"],
            loc=ARTIFACT_PATH / "roc_curve.png",
        )

        artifacts_builder.build_feature_importance_artifact(
            best_model,
            best["X_test"],
            loc=ARTIFACT_PATH / "feature_importance.png",
        )

    logger.info("Evaluation Complete")


if __name__ == "__main__":
    run_evaluation()
