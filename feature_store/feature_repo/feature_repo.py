from datetime import timedelta

import pandas as pd

from feast import (
    Entity,
    FeatureService,
    FeatureView,
    Field,
    FileSource,
    Project,
    PushSource,
    RequestSource,
)
from feast.feature_logging import LoggingConfig
from feast.infra.offline_stores.file_source import FileLoggingDestination
from feast.on_demand_feature_view import on_demand_feature_view
from feast.types import Float32, Float64, Int64

project = Project(name="feature_store", description="time series repo")
unique_id = Entity(name="unique_id", join_keys=["unique_id"])
file_source = FileSource(
    name="interface_traffic_source",
    path="data/train_data.parquet",
    timestamp_field='ds',
)
# Base feature view with the raw 'y' value
interface_traffic_fv = FeatureView(
    name = "interface_traffic",
    entities = [unique_id],
    ttl = timedelta(minutes = 5),
    schema = [
        Field(name = 'y', dtype = Float64),
        # Field(name = 'lag_5min', dtype = Float64),
        # Field(name = 'lag_30min', dtype = Float64),
        # Field(name = 'lag_2h', dtype = Float64),
        # Field(name = 'rolling_mean', dtype = Float64),
        # Field(name = 'rolling_std', dtype = Float64),

    ],
    online = True,
    source = file_source
)



