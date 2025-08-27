#!/bin/bash
set -e

# Lấy đường dẫn tuyệt đối của project
PROJECT_DIR=$(pwd)

# Lưu Airflow home ngay trong project
export AIRFLOW_HOME=$PROJECT_DIR/airflow_home

# Chọn version
AIRFLOW_VERSION=2.9.2
PYTHON_VERSION="$(python --version | cut -d " " -f 2 | cut -d "." -f 1-2)"
CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"

# Cài Airflow
pip install "apache-airflow==${AIRFLOW_VERSION}" --constraint "${CONSTRAINT_URL}"

# Cài dependencies project
pip install -r requirements.txt

# Khởi chạy standalone
airflow standalone
