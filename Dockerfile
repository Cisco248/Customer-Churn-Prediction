# ─── Base Image ───────────────────────────────────────────────────────────
FROM python:3.10-slim

# ─── Environment Variables ────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    AIRFLOW_HOME=/opt/airflow \
    AIRFLOW__CORE__DAGS_FOLDER=/opt/airflow/dags \
    AIRFLOW__CORE__LOAD_EXAMPLES=False \
    AIRFLOW__CORE__EXPOSE_CONFIG=True

# ─── Install System Dependencies ──────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    build-essential \
    libpq-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ─── Create Airflow Home Directory ────────────────────────────────────────────
RUN mkdir -p ${AIRFLOW_HOME}/dags ${AIRFLOW_HOME}/logs ${AIRFLOW_HOME}/plugins

# ─── Copy Requirements ────────────────────────────────────────────────────────
COPY requirements.txt /tmp/
RUN pip install --upgrade pip setuptools wheel

# ─── Install Python Dependencies ──────────────────────────────────────────────
RUN pip install -r /tmp/requirements.txt

# ─── Create App Directory ─────────────────────────────────────────────────────
WORKDIR /app

# ─── Copy Project Files ───────────────────────────────────────────────────────
COPY . /app/

# ─── Copy DAGs to Airflow DAGs folder ─────────────────────────────────────────
RUN cp -r /app/airflow_dags/* ${AIRFLOW_HOME}/dags/

# ─── Initialize Airflow Database ──────────────────────────────────────────────
RUN airflow db init || true

# ─── Create Airflow Admin User (optional, can be set via environment variables) ─
RUN airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin \
    || true

# ─── Expose Ports ─────────────────────────────────────────────────────────────
# Airflow Web UI
EXPOSE 8080
# FastAPI Application
EXPOSE 8000

# ─── Health Check ─────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# ─── Entrypoint Script ────────────────────────────────────────────────────────
COPY docker-entrypoint.sh /app/
RUN chmod +x /app/docker-entrypoint.sh

ENTRYPOINT ["/app/docker-entrypoint.sh"]
