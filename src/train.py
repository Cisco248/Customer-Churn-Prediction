import skops.io as sio
import pandas as pd
import mlflow
import mlflow.sklearn as ml_sk
import mlflow.xgboost as ml_xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from src.config import (
    TRAIN_DATA_PATH,
    TEST_DATA_PATH,
    MODELS_DIR,
    TARGET_COLUMN,
    MLFLOW_TRACKING_URI,
    DAGSHUB_USERNAME,
    DAGSHUB_REPO_NAME,
    DAGSHUB_TOKEN,
)
from src.utils.logger import setup_logger
from src.ml_flow import MLflowConfig
from src.artifacts import ComputeMatrics, Visualization


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

    def setup_model_train(self):

        self.logger.info("Start Model Training")

        self.mlflow

        with mlflow.start_run(run_name=f"{self.model_name}"):

            mlflow.log_params(self.params)

            if self.model == LogisticRegression:
                self.model = LogisticRegression(
                    C=1.0,
                    max_iter=1000,
                    solver="lbfgs",
                    random_state=42,
                )

            elif self.model == RandomForestClassifier:
                self.model = RandomForestClassifier(
                    n_estimators=200,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                )

            elif self.model == XGBClassifier:
                self.model = XGBClassifier(
                    max_depth=8,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    eval_metric="logloss",
                    random_state=42,
                )
            else:
                raise ValueError("Unavailable model type provided.")

            self.logger.info(f"{self.model_name}: Training Started")

            self.model.fit(self.x_train, self.y_train)

            self.y_pred = self.model.predict(self.x_test)
            self.y_prob = self.model.predict_proba(self.x_test)[:, 1]

            self.metrics = ComputeMatrics(self.y_test, self.y_pred, self.y_prob)
            mlflow.log_metrics(self.metrics.init_comput_matrics())

            self.logger.info(f"{self.model_name}: Metrics: {str(self.metrics)}")

            self.artifacts_builder = Visualization(
                self.y_test, self.y_prob, self.y_pred, self.model_name
            )

            if self.model == LogisticRegression:
                self.artifacts_builder.log_confusion_matrix()
                self.artifacts_builder.log_roc_curve(self.model_name)

                ml_sk.log_model(
                    self.model,
                    name=f"train {self.model_name}",
                    params=self.params,
                    registered_model_name=f"{self.model_name.lower()}-churn",
                )

                sio.dump(self.model, self.exp_path)
                mlflow.log_artifact(str(self.exp_path))

            elif self.model == RandomForestClassifier:
                self.artifacts_builder.log_confusion_matrix()
                self.artifacts_builder.log_roc_curve(self.model_name)

                ml_sk.log_model(
                    self.model,
                    name=f"train {self.model_name}",
                    params=self.params,
                    registered_model_name=f"{self.model_name.lower()}-churn",
                )

                sio.dump(self.model, self.exp_path)
                mlflow.log_artifact(str(self.exp_path))

            elif self.model == XGBClassifier:
                self.artifacts_builder.log_confusion_matrix()
                self.artifacts_builder.log_roc_curve(self.model_name)

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

        if not MODELS_DIR:
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            self.logger.info("Building the Models Directory!")

        self.logger.info(f"Training Features: {self.x_train.shape[1]}")

        self.setup_model_train()

        self.logger.info("Training Completed Successfully!")


if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Add project root to path for proper imports
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)

    X_train = train_df.drop(columns=[TARGET_COLUMN]).values
    y_train = train_df[TARGET_COLUMN].values

    X_test = test_df.drop(columns=[TARGET_COLUMN]).values
    y_test = test_df[TARGET_COLUMN].values

    logistic_export_path = str(MODELS_DIR / "logistic_regression.skops")
    logistic_params = {
        "C": 1.0,
        "max_iter": 1000,
        "solver": "lbfgs",
        "random_state": 42,
    }

    random_forest_export_path = str(MODELS_DIR / "random_forest.skops")

    logistic_train = ModelTrainer(
        X_train,
        y_train,
        X_test,
        y_test,
        LogisticRegression,
        logistic_params,
        "Logistic Regression",
        logistic_export_path,
        MLflowConfig(
            MLFLOW_TRACKING_URI,
            DAGSHUB_TOKEN,
            DAGSHUB_USERNAME,
            DAGSHUB_REPO_NAME,
        ),
    )

    logistic_train.setup_model_train()
    logistic_train.run_training()
