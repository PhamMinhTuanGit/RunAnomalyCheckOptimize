import os
import time
import pandas as pd
from feast import FeatureStore
from datetime import datetime
from src.utils.config import load_config
from src.utils.logger import get_logger
from src.models.predict import forecast
from src.data.preprocess import preprocess_data
REALTIME_FOLDER = "data/real_time"
REPO_PATH = "."  # Đường dẫn đến thư mục chứa feature_store.yaml
FEATURE_VIEW_NAME = "driver_hourly_stats"  # Tên FeatureView bạn ingest vào
ON_DEMAND_FEATURE_VIEW_NAME = "transformed_driver_stats"  # Tên on_demand_feature_view

# Theo dõi các file đã xử lý để tránh xử lý lại
processed_files = set()

def 
