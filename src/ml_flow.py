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
            # Validate credentials are properly set (not placeholder values)
            if not self.token or self.token == "ENTER_TOKEN" or "ENTER" in self.token.upper():
                self.logger.error("❌ ===> DAGsHub Token is missing or invalid (placeholder value)")
                raise ValueError("❌ ===> DAGsHub Token is required")
            
            if not self.dagshub_username or self.dagshub_username == "ENTER_USERNAME" or "ENTER" in self.dagshub_username.upper():
                self.logger.error("❌ ===> DAGsHub Username is missing or invalid (placeholder value)")
                raise ValueError("❌ ===> DAGsHub Username is required")
            
            if not self.dagshub_repo_name or self.dagshub_repo_name == "ENTER_REPO_NAME" or "ENTER" in self.dagshub_repo_name.upper():
                self.logger.error("❌ ===> DAGsHub Repo Name is missing or invalid (placeholder value)")
                raise ValueError("❌ ===> DAGsHub Repo Name is required")
            
            if not self.tracking_uri or "ENTER" in self.tracking_uri.upper():
                self.logger.error("❌ ===> MLflow Tracking URI is missing or invalid (placeholder value)")
                raise ValueError("❌ ===> MLflow Tracking URI is required")

            dagshub.init(
                repo_owner=self.dagshub_username,
                repo_name=self.dagshub_repo_name,
                mlflow=True,
            )
            mlflow.set_tracking_uri(self.tracking_uri)
            self.logger.info("✅ ===> DAGsHub authentication successful")
            self.logger.info(f"✅ ===> MLflow: {self.tracking_uri}")

        except Exception as e:
            self.logger.error(f"Error initializing DAGsHub: {e} ===> ❌")
            raise
