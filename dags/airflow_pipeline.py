import os
from dotenv import load_dotenv
import subprocess
from datetime import datetime
from pathlib import Path
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.standard.operators.bash import BashOperator
import dagshub

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
MODELS = ["Logistic_Regression", "Random_Forest", "XGBoost"]


def _build_env() -> dict:
    return {
        **os.environ,
        "DEFAULT_ACCESS_TOKEN": os.environ.get("DEFAULT_ACCESS_TOKEN"),
        "DAGSHUB_USERNAME": os.environ.get("DAGSHUB_USERNAME"),
        "DAGSHUB_REPO_NAME": os.environ.get("DAGSHUB_REPO_NAME"),
        "MLFLOW_TRACKING_URI": os.environ.get("MLFLOW_TRACKING_URI"),
    }


def run_preprocessing():
    subprocess.run(
        [
            "python",
            str(SRC_DIR / "preprocessing.py"),
        ],
        check=True,
    )


def run_training(model_name: str) -> None:

    dagshub.init(
        repo_owner=os.getenv("DAGSHUB_USERNAME"),
        repo_name=os.getenv("DAGSHUB_REPO_NAME"),
    )

    subprocess.run(
        [
            "python",
            str(SRC_DIR / "train.py"),
            "--model_name",
            model_name,
        ],
        env=_build_env(),
        check=True,
    )


def run_evaluation(model_name: str) -> None:
    subprocess.run(
        [
            "python",
            str(SRC_DIR / "evaluate.py"),
            "--model_name",
            model_name,
        ],
        env=_build_env(),
        check=True,
    )


with DAG(
    dag_id="churn_mlops_pipeline",
    description="End-to-end Customer Churn MLOps pipeline with MLflow tracking",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["mlops", "churn", "mlflow", "dagshub"],
    doc_md=__doc__,
) as dag:

    auth = BashOperator(
        task_id="authentication",
        bash_command=f"dagshub login --token {os.getenv('DEFAULT_ACCESS_TOKEN')}",
    )

    task_preprocessing = PythonOperator(
        task_id="preprocessing",
        python_callable=run_preprocessing,
    )

    training_tasks = [
        PythonOperator(
            task_id=f"train_{model_name.lower()}",
            python_callable=run_training,
            op_kwargs={"model_name": model_name},
        )
        for model_name in MODELS
    ]

    evaluation_tasks = [
        PythonOperator(
            task_id=f"evaluate_{model_name.lower()}",
            python_callable=run_evaluation,
            op_kwargs={"model_name": model_name},
        )
        for model_name in MODELS
    ]

    auth >> task_preprocessing >> training_tasks

    for train_task, eval_task in zip(training_tasks, evaluation_tasks):
        train_task >> eval_task
