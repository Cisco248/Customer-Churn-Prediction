"""
Central configuration for the Churn MLOps Project.
All paths, hyperparameters, and settings live here.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─── Project Root ─────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent

# ─── Data Paths ───────────────────────────────────────────────────────────────
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "churn_prediction_data.csv"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train.csv"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test.csv"
PREPROCESSOR_PATH = PROCESSED_DATA_DIR / "preprocessor.joblib"

# ─── Model Paths ──────────────────────────────────────────────────────────────
MODELS_DIR = ROOT_DIR / "models"

# ─── Artifact Paths ───────────────────────────────────────────────────────────
ARTIFACT_PATH = ROOT_DIR / "artifacts"

# ─── MLflow ───────────────────────────────────────────────────────────────────
DAGSHUB_USERNAME = os.getenv("DAGSHUB_USERNAME")
DAGSHUB_REPO_NAME = os.getenv("DAGSHUB_REPO_NAME")
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "https://dagshub.com/Cisco248/customer-churn-prediction.mlflow",
)
MLFLOW_EXPERIMENT_NAME = "customer-churn-prediction"

# ─── Pre-processing ───────────────────────────────────────────────────────────
TARGET_COLUMN = "Churn"
TEST_SIZE = 0.20
RANDOM_STATE = 42

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]

BINARY_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
    "SeniorCitizen",
]

MULTI_FEATURES = [
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaymentMethod",
]

DROP_COLUMNS = ["customerID"]

# ─── Model Hyperparameters ────────────────────────────────────────────────────
LR_PARAMS = {
    "C": 1.0,
    "max_iter": 1000,
    "solver": "lbfgs",
    "random_state": RANDOM_STATE,
}

RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 10,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "use_label_encoder": False,
    "eval_metric": "logloss",
    "random_state": RANDOM_STATE,
}

# Logger Constants

LOG_FILE = "logs/model.log"
FORMAT = "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
