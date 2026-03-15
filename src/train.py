# type: ignore
import argparse
import skops.io as sio
import pandas as pd
import mlflow
import mlflow.sklearn as ml_sk
import mlflow.xgboost as ml_xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import dagshub

from config import (
    TRAIN_DATA_PATH,
    TEST_DATA_PATH,
    LR_EXPORT_PATH,
    LR_PARAMS,
    RF_PARAMS,
    RF_EXPORT_PATH,
    XGB_PARAMS,
    XGB_EXPORT_PATH,
    MODELS_DIR,
    MLFLOW_TRACKING_URI,
    DAGSHUB_USERNAME,
    DAGSHUB_REPO_NAME,
    DAGSHUB_TOKEN,
    X_FEATURE_COLS,
    Y_FEATURE_COLS,
)
from utils.logger import setup_logger
from artifacts import ComputeMatrics, Visualization


class ModelTrainer:
    def __init__(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        model,
        params: dict,
        model_name: str,
        export_path: str,
    ):
        self.x_train = X_train
        self.y_train = y_train
        self.x_test = X_test
        self.y_test = y_test
        self.model = model
        self.params = params
        self.model_name = model_name
        self.exp_path = export_path
        self.logger = setup_logger()

    def _setup_model_train(self):

        self.logger.info("🚀 ===> Training Stage: Started Processing")

        with mlflow.start_run():

            mlflow.log_params(self.params)

            if self.model_name == "Logistic_Regression":
                self.model = LogisticRegression(**self.params)

            elif self.model_name == "Random_Forest":
                self.model = RandomForestClassifier(**self.params)

            elif self.model_name == "XGBoost":
                self.model = XGBClassifier(**self.params)
            else:
                raise ValueError("Unavailable model type provided.")

            self.logger.info(f"{self.model_name}: Training Started ===> ℹ️")

            self.model.fit(self.x_train, self.y_train)

            self.y_pred = self.model.predict(self.x_test)
            self.y_prob = self.model.predict_proba(self.x_test)[:, 1]

            self.metrics = ComputeMatrics(self.y_test, self.y_pred, self.y_prob)
            mlflow.log_metrics(self.metrics.init_comput_matrics())

            self.logger.info(f"{self.model_name}: Metrics: {str(self.metrics)} ===> ℹ️")

            self.artifacts_builder = Visualization(
                self.y_test, self.y_prob, self.y_pred, self.model_name
            )

            if self.model_name == "Logistic_Regression":
                self.artifacts_builder.build_train_confusion_matrix()
                self.artifacts_builder.build_train_roc_curve(self.model_name)

                ml_sk.log_model(
                    self.model,
                    name=f"train {self.model_name}",
                    params=self.params,
                    registered_model_name=f"{self.model_name.lower()}-churn",
                )

                sio.dump(self.model, self.exp_path)
                mlflow.log_artifact(str(self.exp_path))
                mlflow.set_tag("Training Info", f"{self.model_name} for MLOps")

            elif self.model_name == "Random_Forest":
                self.artifacts_builder.build_train_confusion_matrix()
                self.artifacts_builder.build_train_roc_curve(self.model_name)

                ml_sk.log_model(
                    self.model,
                    name=f"train {self.model_name}",
                    params=self.params,
                    registered_model_name=f"{self.model_name.lower()}-churn",
                )

                sio.dump(self.model, self.exp_path)
                mlflow.log_artifact(str(self.exp_path))
                mlflow.set_tag("Training Info", f"{self.model_name} for MLOps")

            elif self.model_name == "XGBoost":
                self.artifacts_builder.build_train_confusion_matrix()
                self.artifacts_builder.build_train_roc_curve(self.model_name)

                ml_xgb.log_model(
                    self.model,
                    name=f"train {self.model_name}",
                    params=self.params,
                    registered_model_name=f"{self.model_name.lower()}-churn",
                )

                sio.dump(self.model, self.exp_path)
                mlflow.log_artifact(str(self.exp_path))
                mlflow.set_tag("Training Info", f"{self.model_name} for MLOps")

            else:
                raise ValueError("Unavailable model type provided.")

    def run_training(self):

        try:
            # Validate credentials are properly set (not placeholder values)
            if (
                not DAGSHUB_TOKEN
                or DAGSHUB_TOKEN == "ENTER_TOKEN"
                or "ENTER" in DAGSHUB_TOKEN.upper()
            ):
                self.logger.error(
                    "❌ ===> DAGsHub Token is missing or invalid (placeholder value)"
                )
                raise ValueError(
                    "❌ ===> DAGsHub Token is required. Set DAGSHUB_TOKEN environment variable with a valid token."
                )

            if (
                not DAGSHUB_USERNAME
                or DAGSHUB_USERNAME == "ENTER_USERNAME"
                or "ENTER" in DAGSHUB_USERNAME.upper()
            ):
                self.logger.error(
                    "❌ ===> DAGsHub Username is missing or invalid (placeholder value)"
                )
                raise ValueError(
                    "❌ ===> DAGsHub Username is required. Set DAGSHUB_USERNAME environment variable."
                )

            if (
                not DAGSHUB_REPO_NAME
                or DAGSHUB_REPO_NAME == "ENTER_REPO_NAME"
                or "ENTER" in DAGSHUB_REPO_NAME.upper()
            ):
                self.logger.error(
                    "❌ ===> DAGsHub Repo Name is missing or invalid (placeholder value)"
                )
                raise ValueError(
                    "❌ ===> DAGsHub Repo Name is required. Set DAGSHUB_REPO_NAME environment variable."
                )

            if not MLFLOW_TRACKING_URI or "ENTER" in MLFLOW_TRACKING_URI.upper():
                self.logger.error(
                    "❌ ===> MLflow Tracking URI is missing or invalid (placeholder value)"
                )
                raise ValueError(
                    "❌ ===> MLflow Tracking URI is required. Set MLFLOW_TRACKING_URI environment variable."
                )

            dagshub.init(
                repo_owner=DAGSHUB_USERNAME,
                repo_name=DAGSHUB_REPO_NAME,
                mlflow=True,
            )
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            self.logger.info("✅ ===> DAGsHub authentication successful")
            self.logger.info(f"✅ ===> MLflow: {MLFLOW_TRACKING_URI}")

        except Exception as e:
            self.logger.error(f"Error initializing DAGsHub: {e} ===> ❌")
            raise

        mlflow.set_experiment("churn-training")

        if not MODELS_DIR.exists():
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"⚠️ ===> Models directory not found! {MODELS_DIR}")

        self.logger.info("Models directory build successfully! ===> ℹ️")

        self.logger.info(f"Training Features: {self.x_train.shape[1]} ===> ℹ️")

        self._setup_model_train()

        self.logger.info("✅ ===> Training Completed Successfully!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Choose: Logistic_Regression, Random_Forest, or XGBoost",
    )
    args = parser.parse_args()

    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)

    X_train = train_df[X_FEATURE_COLS].values
    y_train = train_df[Y_FEATURE_COLS].values

    X_test = test_df[X_FEATURE_COLS].values
    y_test = test_df[Y_FEATURE_COLS].values

    if args.model_name == "Logistic_Regression":
        trainer = ModelTrainer(
            X_train,
            y_train,
            X_test,
            y_test,
            LogisticRegression,
            LR_PARAMS,
            "Logistic_Regression",
            LR_EXPORT_PATH,
        )
    elif args.model_name == "Random_Forest":
        trainer = ModelTrainer(
            X_train,
            y_train,
            X_test,
            y_test,
            RandomForestClassifier,
            RF_PARAMS,
            "Random_Forest",
            RF_EXPORT_PATH,
        )
    elif args.model_name == "XGBoost":
        trainer = ModelTrainer(
            X_train,
            y_train,
            X_test,
            y_test,
            XGBClassifier,
            XGB_PARAMS,
            "XGBoost",
            XGB_EXPORT_PATH,
        )
    else:
        raise ValueError(f"Model {args.model_name} not recognized!")

    trainer.run_training()
