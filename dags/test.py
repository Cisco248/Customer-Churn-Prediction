import os
from pathlib import Path
from dotenv import load_dotenv

# load_dotenv()


ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"


def _build_env() -> dict:
    return {
        "DAGSHUB_TOKEN": os.environ.get("DAGSHUB_TOKEN", ""),
        "DAGSHUB_USERNAME": os.environ.get("DAGSHUB_USERNAME", ""),
        "DAGSHUB_REPO_NAME": os.environ.get("DAGSHUB_REPO_NAME", ""),
        "MLFLOW_TRACKING_URI": os.environ.get("MLFLOW_TRACKING_URI", ""),
    }


print(_build_env())
