from datetime import timedelta
from feast import Entity, FeatureView, FileSource, Field
from feast.types import Int32, Float32
import os
# Entity: định danh cho mỗi chuỗi time-series
time_series = Entity(
    name="unique_id",
    join_keys=["unique_id"],
    description="ID cho từng chuỗi thời gian",
)

# Source: trỏ tới file CSV đã được xử lý
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

time_series_source = FileSource(
    path=os.path.join(REPO_ROOT, "data/processed/train_data.csv"),
    timestamp_field="ds",
)
# Feature View
time_series_fv = FeatureView(
    name="time_series_features",
    entities=[time_series],
    ttl=timedelta(days=365),
    schema=[
        Field(name="y", dtype=Float32),
        Field(name="day_of_week", dtype=Int32),
        Field(name="hour", dtype=Int32),
        Field(name="lag1", dtype=Float32),
        Field(name="lag7", dtype=Float32),
    ],
    online=True,
    source=time_series_source,
)
