# End to End Customer Churn Prediction MLOps Project

## Installaion Guidence

### 1. Linux Operating System Configuration

#### 1.1 Local Container Setup

> Do NOT use your Windows `.venv` — it won't work in Linux. Create a separate one.

##### 1.1.1 | Create Local Virtual Environment

```bash
python3 -m venv .<ANY_NAME>
```

##### 1.1.2 | Activate Created Virtual Environment

```bash
source .<ANY_NAME>/bin/activate
```

##### 1.1.3 | Upgrade Pip in Selected `.venv`

```bash
pip install --upgrade pip
```

##### 1.1.4 | Install Airflow Constant for the `/Directory`

```bash
AIRFLOW_VERSION=<ENTER_VERSION>    # ---> Recommended Version for [Airflow=2.10.4]
PYTHON_VERSION=<ENTER_VERSION>     # ---> Recommended Version for [PYTHON_VERSION=3.11]
CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"
```

##### 1.1.5 | Install All Libraries and Dependencies Using `requirements.txt`

```bash
pip install -r requirements.txt
pip install "apache-airflow==${AIRFLOW_VERSION}" --constraint "${CONSTRAINT_URL}"
```

##### 1.1.6 | Initialize Airflow

```bash
export AIRFLOW_HOME=~/airflow
airflow db init
```

##### 1.1.7 | Create New User for Airflow `admin:admin`

```bash
airflow users create \ 
  --username admin \            # ---> Defaulf admin
  --password admin \            # ---> Defaulf admin
  --firstname Admin \           # ---> Defaulf admin
  --lastname User \             # ---> Defaulf admin
  --role Admin \                # ---> Defaulf admin
  --email admin@example.com     # ---> Defaulf admin
```

##### OR

```bash
airflow users create \ 
  --username <ANY_NAME> \            # ---> Defaulf ANY_NAME
  --password <ANY_PASSWORD> \        # ---> Defaulf ANY_PASSWORD
  --firstname <ANY_FIRST_NAME> \     # ---> Defaulf ANY_FIRST_NAME
  --lastname <ANY_LAST_NAME> \       # ---> Defaulf ANY_LAST_NAME
  --role <ANY_ROLE> \                # ---> Defaulf ANY_ROLE
  --email <ANY_EMAIL>                # ---> Defaulf ANY_EMAIL
```

##### 1.1.8 | Configure the `airflow.cfg`

> Edit the Airflow config to point to your project's DAGs:

##### 1. Hard-Code Method

```bash
nano ~/airflow/airflow.cfg
```

> Find the `[core]` section and change `dags_folder`:

```ini
[core]
dags_folder = /mnt/c/Projects/churn-mlops-project - test/airflow_dags
```

>Also set `load_examples = False` to hide sample DAGs:

```ini
load_examples = False
```

> Save and exit (`Ctrl+O`, `Enter`, `Ctrl+X`).

##### 2. CLI Method

> Make Custom using this Codelines

```bash
sed -i 's|^dag = .*|dag = <YOUR_LOCAL_DIRECTORY> - test/dag|' ~/airflow/airflow.cfg
sed -i 's|^load_examples = .*|load_examples = False|' ~/airflow/airflow.cfg
grep -E "dag|load_examples" ~/airflow/airflow.cfg
```

##### 1.1.9 | Set Environment Variables for MLflow/DAGsHub

> Once add this commands to the directory, It will remove automatically.

```bash
export MLFLOW_TRACKING_URI="https://dagshub.com/<DAGSHUB_USERNAME>/<DAGSHUB_REPO_NAME>.mlflow"
export MLFLOW_TRACKING_USERNAME="<DAGSHUB_USERNAME>"
export MLFLOW_TRACKING_PASSWORD="<DAGSHUB_TOKEN>"
```

> To make these permanent, add them to `~/.bashrc`:

```bash
echo 'export MLFLOW_TRACKING_URI="https://dagshub.com/<DAGSHUB_USERNAME>/<DAGSHUB_REPO_NAME>.mlflow"' >> ~/.bashrc
echo 'export MLFLOW_TRACKING_USERNAME="<DAGSHUB_USERNAME>"' >> ~/.bashrc
echo 'export MLFLOW_TRACKING_PASSWORD="<DAGSHUB_TOKEN>"' >> ~/.bashrc
source ~/.bashrc
```

##### 1.1.10 | Start Airflow

> In this Process, `2 CMD, Terminals or Powershell Needed!`

##### Start Terminal 1 — Web Server

> Open a New Terminal, It's Provides Airflow UI:

```bash
source .<ANY_NAME>/bin/activate
airflow webserver --port 8080
```

##### Terminal 2 — Scheduler

> Open a New Terminal, It's Provides Scheduler Service in Airflow

```bash
cd <YOUR_LOCAL_DIRECTORY>-\ test
source .<ANY_NAME>/bin/activate
export AIRFLOW_HOME=~/airflow
airflow scheduler
```

##### 1.1.11 | Access Airflow UI

> Open Your Browser, Copy Link and Go:

```link
http://localhost:8080
```

> Enter the Credentails in the UI Login

Username: `admin` | Password: `admin`

##### 1.1.12 | Trigger the DAG

1. In the Airflow UI, find `churn_prediction_pipeline` in the DAGs list
2. Toggle the DAG **ON** (switch on the left)
3. Click the **▶ Trigger DAG** button (play icon on the right)
4. Watch the tasks execute in sequence (click on the DAG name → Graph view)

---
