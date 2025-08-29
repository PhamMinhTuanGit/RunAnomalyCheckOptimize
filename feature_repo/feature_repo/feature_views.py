from datetime import timedelta
from feast import Entity, FeatureView, FileSource, Field
from feast.types import Int32, Float32
import os

import yaml

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

   
data_config = load_config("config.yaml")


time_series = Entity(
    name="unique_id",
    join_keys=["unique_id"],
    description="ID cho từng chuỗi thời gian",
)

# Source: trỏ tới file CSV đã được xử lý
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

time_series_source = FileSource(
    path=os.path.join(REPO_ROOT, 'data/processed/train_data.parquet'),
    timestamp_field="ds",
)

time_series_fv = FeatureView(
    name="time_series_features",
    entities=[time_series],
    ttl=timedelta(days=365),
    schema=[
        Field(name="y", dtype=Float32),
        Field(name="rolling_mean", dtype=Float32),
        Field(name="rolling_std", dtype=Float32),
        Field(name="lag_5min", dtype=Float32),
        Field(name="lag_30min", dtype=Float32),
        Field(name="lag_2h", dtype=Float32),
        
    ],
    online=True,
    source=time_series_source,
)
