FROM apache/airflow:3.1.8

# Copy the requirements.txt file to the container
COPY requirements.txt ./

# Delete the airflow and psycopg2-binary packages from requirements.txt and install the remaining packages
RUN sed -i -e '/apache-airflow/d' -e '/psycopg2-binary/d' requirements.txt && pip install --no-cache-dir -r requirements.txt