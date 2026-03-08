import skops.io as sio
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn as ml_sk
import mlflow.xgboost as ml_xgb

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)
from xgboost import XGBClassifier

from config import (
    TRAIN_DATA_PATH,
    TEST_DATA_PATH,
    MODELS_DIR,
    TARGET_COLUMN,
    ARTIFACT_PATH,
)

from utils.logger import setup_logger
from ml_flow import MLflowConfig

logger = setup_logger()


# ─────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    return {
        "Accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "Precision": round(float(precision_score(y_true, y_pred)), 4),
        "Recall": round(float(recall_score(y_true, y_pred)), 4),
        "F1": round(float(f1_score(y_true, y_pred)), 4),
        "Roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
    }


# ─────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────


def log_confusion_matrix(y_true, y_pred, run_name: str):

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Churn", "Churn"])
    ax.set_yticklabels(["No Churn", "Churn"])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix – {run_name}")

    plt.colorbar(im)

    path = ARTIFACT_PATH / f"cm_{run_name.replace(' ', '_')}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

    mlflow.log_artifact(str(path))


def log_roc_curve(y_true, y_prob, run_name: str):

    fpr, tpr, _ = roc_curve(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], "k--")

    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(f"ROC Curve – {run_name}")

    path = ARTIFACT_PATH / f"roc_{run_name.replace(' ', '_')}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

    mlflow.log_artifact(str(path))


# ─────────────────────────────────────────────────────────────
# Logistic Regression
# ─────────────────────────────────────────────────────────────


def logistic_regression_train_model(X_train, y_train, X_test, y_test):

    logger.info("\nTraining: Logistic Regression")

    with mlflow.start_run(run_name="Logistic Regression"):

        params = {
            "C": 1.0,
            "max_iter": 1000,
            "solver": "lbfgs",
            "random_state": 42,
        }

        mlflow.log_params(params)

        model = LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver="lbfgs",
            random_state=42,
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = compute_metrics(y_test, y_pred, y_prob)
        mlflow.log_metrics(metrics)

        logger.info(f"Logistic Regression Metrics: {str(metrics)}")

        log_confusion_matrix(y_test, y_pred, "Logistic Regression")
        log_roc_curve(y_test, y_prob, "Logistic Regression")

        ml_sk.log_model(
            model, name="model", params=params, registered_model_name="lr-churn"
        )

        model_path = MODELS_DIR / "logistic_regression.skops"
        sio.dump(model, model_path)
        mlflow.log_artifact(str(model_path))


# ─────────────────────────────────────────────────────────────
# Random Forest
# ─────────────────────────────────────────────────────────────


def random_forest_train_model(X_train, y_train, X_test, y_test):

    logger.info("\nTraining: Random Forest")

    with mlflow.start_run(run_name="Random Forest"):

        params = {
            "n_estimators": 200,
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "random_state": 42,
            "n_jobs": -1,
        }

        mlflow.log_params(params)

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = compute_metrics(y_test, y_pred, y_prob)
        mlflow.log_metrics(metrics)

        logger.info(f"Random Forest Metrics: {str(metrics)}")

        log_confusion_matrix(y_test, y_pred, "Random Forest")
        log_roc_curve(y_test, y_prob, "Random Forest")

        ml_sk.log_model(
            model, name="model", params=params, registered_model_name="rf-churn"
        )

        model_path = MODELS_DIR / "random_forest.skops"
        sio.dump(model, model_path)
        mlflow.log_artifact(str(model_path))


# ─────────────────────────────────────────────────────────────
# XGBoost
# ─────────────────────────────────────────────────────────────


def xg_boost_train_model(X_train, y_train, X_test, y_test):

    logger.info("\nTraining: XGBoost")

    with mlflow.start_run(run_name="XGBoost"):

        params = {
            "max_depth": 8,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "eval_metric": "logloss",
            "random_state": 42,
        }

        mlflow.log_params(params)

        model = XGBClassifier(
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = compute_metrics(y_test, y_pred, y_prob)
        mlflow.log_metrics(metrics)

        logger.info(f"XGBoost Metrics: {str(metrics)}")

        log_confusion_matrix(y_test, y_pred, "XGBoost")
        log_roc_curve(y_test, y_prob, "XGBoost")

        ml_xgb.log_model(
            model, name="model", params=params, registered_model_name="xgb-churn"
        )

        model_path = MODELS_DIR / "xgboost.skops"
        sio.dump(model, model_path)
        mlflow.log_artifact(str(model_path))


# ─────────────────────────────────────────────────────────────
# Main Training Runner
# ─────────────────────────────────────────────────────────────


def run_training():

    flow = MLflowConfig()
    flow.initialize()

    if not MODELS_DIR:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("Building the Models Directory!")

    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)

    X_train = train_df.drop(columns=[TARGET_COLUMN]).values
    y_train = train_df[TARGET_COLUMN].values

    X_test = test_df.drop(columns=[TARGET_COLUMN]).values
    y_test = test_df[TARGET_COLUMN].values

    logger.info(f"Training Features: {X_train.shape[1]}")

    logistic_regression_train_model(X_train, y_train, X_test, y_test)
    random_forest_train_model(X_train, y_train, X_test, y_test)
    xg_boost_train_model(X_train, y_train, X_test, y_test)

    logger.info("Training Completed Successfully!")


if __name__ == "__main__":
    run_training()
