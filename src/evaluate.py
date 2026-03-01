# type: ignore
import joblib
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
from ml_flow import setup_mlflow

logger = setup_logger()


def evaluate_model(model_name: str, model_path):

    logger.info(f"\n{'='*50}\n{model_name}\n{'='*50}")

    model = joblib.load(model_path)

    test_df = pd.read_csv(TEST_DATA_PATH)
    X_test = test_df.drop(columns=[TARGET_COLUMN])
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

    logger.info(
        "\nClassification Report:\n"
        + classification_report(y_test, y_pred, target_names=["No Churn", "Churn"])
    )

    return model, metrics, y_test, y_prob, X_test


def run_evaluation():

    setup_mlflow()

    logger.info("Running Final Evaluation on All Models")

    models = {
        "Logistic_Regression": MODELS_DIR / "logistic_regression.joblib",
        "Random_Forest": MODELS_DIR / "random_forest.joblib",
        "XGBoost": MODELS_DIR / "xgboost.joblib",
    }

    best_auc = 0
    best_model_name = None
    best_results = None

    for name, path in models.items():

        model, metrics, y_test, y_prob, X_test = evaluate_model(name, path)

        if metrics["roc_auc"] > best_auc:
            best_auc = metrics["roc_auc"]
            best_model_name = name
            best_results = (model, metrics, y_test, y_prob, X_test)

    logger.info(f"\nBest Model: {best_model_name} | ROC-AUC: {best_auc}")

    # ===== MLflow Logging =====
    model, metrics, y_test, y_prob, X_test = best_results

    with mlflow.start_run(run_name="customer-churn-prediction"):

        mlflow.set_tag("stage", "evaluation")
        mlflow.set_tag("best_model", best_model_name)

        mlflow.log_metrics(metrics)

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr)
        ax.plot([0, 1], [0, 1], "k--")
        ax.set_title("ROC Curve")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")

        roc_path = ARTIFACT_PATH / "roc_curve.png"
        fig.savefig(roc_path)
        plt.close(fig)

        mlflow.log_artifact(str(roc_path))

        # Feature Importance (Tree models only)
        if hasattr(model, "feature_importances_"):

            importance = (
                pd.DataFrame(
                    {
                        "feature": X_test.columns,
                        "importance": model.feature_importances_,
                    }
                )
                .sort_values("importance", ascending=False)
                .head(20)
            )

            fig, ax = plt.subplots()
            ax.barh(importance["feature"], importance["importance"])
            ax.set_title("Top 20 Feature Importances")

            fi_path = ARTIFACT_PATH / "feature_importance.png"
            fig.savefig(fi_path)
            plt.close(fig)

            mlflow.log_artifact(str(fi_path))

    logger.info("Evaluation Complete")


if __name__ == "__main__":
    run_evaluation()
