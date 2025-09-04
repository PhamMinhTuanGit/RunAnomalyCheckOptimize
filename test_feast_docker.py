from feast import FeatureStore
import mlflow
from src.models.train import train_patchtst, train_nhits
from src.utils.logger import get_logger
from src.utils.config import load_config
import pandas as pd
import logging
import os

def test_training():
    config = load_config("config.yaml")
    df = pd.read_parquet(config["data"]["processed_data_parquet_path"])
    split_point = int(len(df) * 0.8)
    train_df = df.iloc[:split_point]
    test_df = df.iloc[split_point:]
    print(f"Train set size: {len(train_df)}, Test set size: {len(test_df)}")
    
    model, n_params, score = train_nhits(train_df, test_df, config)
    print(f"Model trained with {n_params} parameters and MAPE score: {score}")


if __name__ == "__main__":
    test_training()
