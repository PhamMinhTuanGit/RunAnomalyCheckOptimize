from feast import FeatureStore
import mlflow
from src.models.train import train_nhits
from src.utils.logger import get_logger
from src.utils.config import load_config
import pandas as pd
import logging
import os

def test_training():
    try:
        config = load_config("config.yaml")
        store = FeatureStore(repo_path=config['data']['feature_store_path'])
        entity_df = pd.read_parquet(config["data"]["processed_data_parquet_path"])[['unique_id', 'ds']]
        training_df = store.get_historical_features(
            entity_df = entity_df[["unique_id", "ds"]],
            features = ["time_series_fv:y"]
        ).to_df()
        print(training_df.head())
    except:
        print("error")    

if __name__ == "__main__":
    test_training()
