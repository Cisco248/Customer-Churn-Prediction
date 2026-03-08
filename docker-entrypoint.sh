#!/bin/bash
set -e

# Initialize Airflow database
airflow db init

# Create admin user if it doesn't exist
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin \
    2>/dev/null || true

# Parse command line arguments
if [ "$1" = "airflow-webserver" ]; then
    echo "Starting Airflow Web Server..."
    exec airflow webserver --port 8080

elif [ "$1" = "airflow-scheduler" ]; then
    echo "Starting Airflow Scheduler..."
    exec airflow scheduler

elif [ "$1" = "api" ]; then
    echo "Starting FastAPI..."
    cd /app
    exec uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

elif [ "$1" = "bash" ] || [ "$1" = "sh" ]; then
    exec "$@"
    
else
    # Default: start Airflow webserver
    echo "Starting Airflow Web Server (default)..."
    exec airflow webserver --port 8080
fi
