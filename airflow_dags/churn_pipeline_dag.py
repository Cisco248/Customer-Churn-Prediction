from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
import subprocess


def run_preprocessing():
    subprocess.run(["python", "src/preprocessing.py"])


def run_training():
    subprocess.run(["python", "src/train.py"])


def run_evaluation():
    subprocess.run(["python", "src/evaluate.py"])


with DAG("churn_pipeline", catchup=False) as dag:

    task1 = PythonOperator(task_id="preprocessing", python_callable=run_preprocessing)

    task2 = PythonOperator(task_id="training", python_callable=run_training)

    task3 = PythonOperator(task_id="evaluation", python_callable=run_evaluation)

    task1 >> task2 >> task3
