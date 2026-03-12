import dagshub
import mlflow
from utils.logger import setup_logger


class MLflowConfig:

    def __init__(
        self, uri: str, token: str | None, username: str | None, repo_name: str | None
    ) -> None:
        self.tracking_uri = uri
        self.token = token
        self.dagshub_username = username
        self.dagshub_repo_name = repo_name
        self.logger = setup_logger()

        try:
            if not self.token:
                self.logger.info("❌ ===> DAGsHub authentication unsuccessful")

            self.logger.info("✅ ===> DAGsHub authentication successful")
            dagshub.init(
                repo_owner=self.dagshub_username,
                repo_name=self.dagshub_repo_name,
                mlflow=True,
            )
            mlflow.set_tracking_uri(self.tracking_uri)
            self.logger.info(f"✅ ===> MLflow: {self.tracking_uri}")

        except Exception as e:
            self.logger.error(f"Error initializing DAGsHub: {e} ===> ❌")

    def init_experiment(self, name: str | None):
        self.experiment_name = name
        mlflow.set_experiment(self.experiment_name)
