from feast import FeatureStore
from src.models.train import train_nhits
from src.utils.logger import get_logger
from src.utils.config import load_config
import pandas as pd

logger = get_logger("Pipeline")

def run_pipeline():
    config = load_config("config.yaml")

    # 1. Load entity_df từ file parquet đã xử lý
    entity_df = pd.read_parquet(config["data"]["processed_data_parquet_path"])

    # Giữ lại cột entity key và timestamp (giả sử là unique_id và ds)
    entity_df = entity_df[["unique_id", "ds"]]

    # 2. Khởi tạo Feast FeatureStore
    store = FeatureStore(repo_path=config['data']['feature_store_path'])  # sửa path repo Feast của bạn

    # 3. Lấy dữ liệu feature lịch sử từ Feast
    feature_list = [
        "time_series_features:y",
        "time_series_features:day_of_week",
        "time_series_features:hour",
        "time_series_features:lag1",
        "time_series_features:lag7",
    ]

    df = store.get_historical_features(
        entity_df=entity_df,
        features=feature_list
    ).to_df()

    # 4. Split train/test
    train_size = int(len(df) * config["data"]["split_ratio"])
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]

    # 5. Train model
    model, score = train_nhits(train_df, test_df, config)
    logger.info(f"Pipeline completed. Final Test MAPE = {score:.2f}")

if __name__ == "__main__":
    run_pipeline()
