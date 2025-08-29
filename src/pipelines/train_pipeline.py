from feast import FeatureStore
from src.models.train import train_nbeatsx
from src.utils.logger import get_logger
from src.utils.config import load_config
import pandas as pd

logger = get_logger("Pipeline")

def run_pipeline():
    config = load_config("config.yaml")
    store = FeatureStore(repo_path=config['data']['feature_store_path'])  # sửa path repo Feast của bạn
    entity_df = pd.read_parquet(config["data"]["processed_data_parquet_path"])[['unique_id', 'ds']]
    # 3. Lấy dữ liệu feature lịch sử từ Feast
    feature_list = [
        "time_series_features:y",
        "time_series_features:lag_5min",
        "time_series_features:lag_30min",
        "time_series_features:rolling_mean",
        "time_series_features:rolling_std",
        "time_series_features:lag_2h"
    ]

    df = store.get_historical_features(
        entity_df=entity_df[["unique_id", "ds"]],
        features=feature_list
    ).to_df()

    # 4. Split train/test
    train_size = int(len(df) * config["data"]["split_ratio"])
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]

    # 5. Train model
    model, score = train_nbeatsx(train_df, test_df, config)
    logger.info(f"Pipeline completed. Final Test MAPE = {score:.2f}")

if __name__ == "__main__":
    run_pipeline()
