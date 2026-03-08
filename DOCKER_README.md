# Docker Setup Guide for Customer Churn Prediction with Airflow

## Overview

This Docker setup includes:
- **Airflow Webserver**: UI for managing and monitoring DAGs (Port 8080)
- **Airflow Scheduler**: Automatically runs the ML pipeline DAGs
- **FastAPI**: REST API for serving the trained churn prediction model (Port 8000)
- **PostgreSQL**: Database for Airflow metadata

## Prerequisites

- Docker Desktop installed and running
- Docker Compose installed

## Quick Start

### 1. Build the Docker Image

```bash
docker-compose build
```

### 2. Start All Services

```bash
docker-compose up -d
```

### 3. Access the Services

- **Airflow Web UI**: http://localhost:8080
  - Username: `admin`
  - Password: `admin`

- **FastAPI Docs**: http://localhost:8000/docs
- **FastAPI Health**: http://localhost:8000/

## Individual Docker Commands

### Build Image

```bash
docker build -t churn-prediction:latest .
```

### Run Airflow Webserver Only

```bash
docker run -p 8080:8080 -v $(pwd)/airflow_dags:/opt/airflow/dags -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models --name airflow-web churn-prediction:latest airflow-webserver
```

### Run Airflow Scheduler Only

```bash
docker run -v $(pwd)/airflow_dags:/opt/airflow/dags -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models -v $(pwd)/src:/app/src --name airflow-scheduler churn-prediction:latest airflow-scheduler
```

### Run FastAPI Only

```bash
docker run -p 8000:8000 -v $(pwd)/models:/app/models -v $(pwd)/data:/app/data --name churn-api churn-prediction:latest api
```

## Docker Compose Commands

### Start Services

```bash
docker-compose up -d
```

### Stop Services

```bash
docker-compose down
```

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f airflow-webserver
docker-compose logs -f airflow-scheduler
docker-compose logs -f api
```

### Rebuild (after code changes)

```bash
docker-compose build --no-cache
docker-compose up -d
```

### Access Service Shells

```bash
# Airflow webserver shell
docker-compose exec airflow-webserver bash

# API shell
docker-compose exec api bash
```

## Airflow DAG Management

### Trigger DAG Manually

1. Open http://localhost:8080
2. Locate the `churn_pipeline` DAG
3. Click the play button to trigger the DAG

### View DAG Runs

- Monitor progress in the Airflow Web UI
- Check logs for each task in the DAG

### Update DAGs

1. Modify files in `./airflow_dags/`
2. Restart services (Airflow auto-detects changes within 30 seconds by default)

```bash
docker-compose restart airflow-scheduler
```

## Troubleshooting

### Postgres Connection Issues

```bash
# Check postgres container health
docker-compose ps

# Restart postgres
docker-compose restart postgres
```

### Airflow Won't Start

```bash
# Check logs
docker-compose logs airflow-webserver

# Re-initialize database
docker-compose exec airflow-webserver airflow db reset
```

### API Not Responding

```bash
# Check if models exist
docker-compose exec api ls models/

# Test API health
curl http://localhost:8000/
```

### Volume Permissions (Linux)

If you encounter permission errors, run:

```bash
sudo chown -R $USER:$USER logs/ airflow_dags/ data/
```

## Environment Variables

You can customize behavior by creating a `.env` file in the project root:

```env
AIRFLOW__CORE__MAX_ACTIVE_TASKS_PER_DAG=16
AIRFLOW__SCHEDULER__DAG_DIR_LIST_INTERVAL=300
AIRFLOW__CORE__PARALLELISM=32
```

## Cleaning Up

### Remove All Containers and Volumes

```bash
docker-compose down -v
```

### Prune Unused Docker Resources

```bash
docker system prune -a
```

## Production Considerations

1. **Change Default Airflow Credentials**: Update the `docker-entrypoint.sh` script
2. **Use Environment Variables**: Store sensitive data in `.env` files
3. **Use Kubernetes Executor**: For production, consider using KubernetesExecutor instead of LocalExecutor
4. **External Database**: Use managed PostgreSQL (AWS RDS, etc.) instead of containerized Postgres
5. **Volume Management**: Use persistent volumes or external storage for logs and data

## Notes

- The FastAPI application runs on `0.0.0.0:8000` inside the container
- Airflow DAGs are automatically synced from `./airflow_dags` directory
- All logs are stored in `./logs/` directory
- Model files are mounted from `./models/` directory for easy updates
