import os
from pathlib import Path
from dotenv import load_dotenv
from xgboost import Booster, XGBClassifier

load_dotenv()

# ─── Project Root ─────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent

# ─── Data Paths ───────────────────────────────────────────────────────────────
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "churn_prediction_data.csv"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
GENERATED_DATA_DIR = ROOT_DIR / "data" / "generated"
TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train.csv"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test.csv"

# ─── Models ──────────────────────────────────────────────────────────────
MODELS_DIR = ROOT_DIR / "models"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessing"
REPORT_TITLE = "\nClassification Report:"
TARGET_NAMES = ["No Churn", "Churn"]
TRUST_LIST = [Booster, XGBClassifier]


# ─── MLflow ───────────────────────────────────────────────────────────────────
DAGSHUB_USERNAME = os.getenv("DAGSHUB_USERNAME", "").strip()
DAGSHUB_REPO_NAME = os.getenv("DAGSHUB_REPO_NAME", "").strip()
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN", "").strip()

# Construct MLFLOW_TRACKING_URI from credentials or use explicit override
if DAGSHUB_USERNAME and DAGSHUB_REPO_NAME and DAGSHUB_USERNAME != "ENTER_USERNAME":
    _default_uri = f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO_NAME}/mlflow"
else:
    _default_uri = "https://dagshub.com/ENTER_USERNAME/ENTER_REPO_NAME/mlflow"

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", _default_uri)

MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME")

# ─── Pre-processing ───────────────────────────────────────────────────────────
TEST_SIZE = 0.2
RANDOM_STATE = 42

TARGET_COLUMN = "Churn"

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

REQUIRED_COLS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    TARGET_COLUMN,
]

X_FEATURE_COLS = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "MultipleLines_No",
    "MultipleLines_No phone service",
    "MultipleLines_Yes",
    "InternetService_DSL",
    "InternetService_Fiber optic",
    "InternetService_No",
    "OnlineSecurity_No",
    "OnlineSecurity_No internet service",
    "OnlineSecurity_Yes",
    "OnlineBackup_No",
    "OnlineBackup_No internet service",
    "OnlineBackup_Yes",
    "DeviceProtection_No",
    "DeviceProtection_No internet service",
    "DeviceProtection_Yes",
    "TechSupport_No",
    "TechSupport_No internet service",
    "TechSupport_Yes",
    "StreamingTV_No",
    "StreamingTV_No internet service",
    "StreamingTV_Yes",
    "StreamingMovies_No",
    "StreamingMovies_No internet service",
    "StreamingMovies_Yes",
    "Contract_Month-to-month",
    "Contract_One year",
    "Contract_Two year",
    "PaymentMethod_Bank transfer (automatic)",
    "PaymentMethod_Credit card (automatic)",
    "PaymentMethod_Electronic check",
    "PaymentMethod_Mailed check",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
]

Y_FEATURE_COLS = ["Churn"]


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
    "eval_metric": "logloss",
    "random_state": RANDOM_STATE,
}

# Model locations for evaluation

LR_EXPORT_PATH = MODELS_DIR / "Logistic_Regression.skops"
RF_EXPORT_PATH = MODELS_DIR / "Random_Forest.skops"
XGB_EXPORT_PATH = MODELS_DIR / "XGBoost.skops"

# Logger Constants

LOG_FILE = "debug.log"
FORMAT = "%(asctime)s [%(levelname)s] %(message)s"

# Evaluation Parameters

BEST_AUC = 0
BEST_MODEL_NAME = None
BEST_RESULTS = None

# ─── Artifact Paths ───────────────────────────────────────────────────────────
ARTIFACT_PATH = ROOT_DIR / "artifacts"

ARTIFACT_TRAIN_PATH = ARTIFACT_PATH / "train"
ARTIFACT_EVALUATION_PATH = ARTIFACT_PATH / "evaluation"
