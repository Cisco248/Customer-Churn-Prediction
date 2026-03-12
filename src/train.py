import argparse
import skops.io as sio
import pandas as pd
import mlflow
import mlflow.sklearn as ml_sk
import mlflow.xgboost as ml_xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

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
from ml_flow import MLflowConfig
from artifacts import ComputeMatrics, Visualization  # type: ignore


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
        mlflow_config: MLflowConfig,
    ):
        self.mlflow = mlflow_config
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

        self.mlflow

        with mlflow.start_run(run_name=f"{self.model_name}"):

            mlflow.log_params(self.params)

            if self.model == LogisticRegression:
                self.model = LogisticRegression(**self.params)

            elif self.model == RandomForestClassifier:
                self.model = RandomForestClassifier(**self.params)

            elif self.model == XGBClassifier:
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

            if self.model_name == "Logistic Regression":
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

            elif self.model_name == "Random Forest":
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

            else:
                raise ValueError("Unavailable model type provided.")

    def run_training(self):

        self.mlflow

        if MODELS_DIR:
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            self.logger.info("⚠️ ===> Models directory already exists!")

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
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

    ml_config = MLflowConfig(
        MLFLOW_TRACKING_URI,
        DAGSHUB_TOKEN,
        DAGSHUB_USERNAME,
        DAGSHUB_REPO_NAME,
    )

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
            "Logistic Regression",
            LR_EXPORT_PATH,
            ml_config,
        )
    elif args.model_name == "Random_Forest":
        trainer = ModelTrainer(
            X_train,
            y_train,
            X_test,
            y_test,
            RandomForestClassifier,
            RF_PARAMS,
            "Random Forest",
            RF_EXPORT_PATH,
            ml_config,
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
            ml_config,
        )
    else:
        raise ValueError(f"Model {args.model_name} not recognized!")

    trainer.run_training()
