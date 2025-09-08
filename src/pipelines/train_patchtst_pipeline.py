from feast import FeatureStore
import mlflow
from src.models.train import train_patchtst
from src.utils.logger import get_logger
from src.utils.config import load_config
import pandas as pd
import logging
import os

def run_pipeline():
    config = load_config("config.yaml")

    # Tạo thư mục logs nếu chưa tồn tại
    log_dir = os.path.dirname(config["log"]["training_log"])
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    # Thiết lập logging để ghi ra file và console
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(config["log"]["training_log"]),
            logging.StreamHandler()
        ]
    )
    logger = get_logger("Pipeline")

    mlflow.set_experiment("PatchTST Experiment")
    with mlflow.start_run(run_name=config["experiment"]["run_name"]):
        mlflow.log_artifact("config.yaml")
        store = FeatureStore(repo_path=config['data']['feature_store_path'])  # sửa path repo Feast của bạn
        entity_df = pd.read_parquet(config["data"]["processed_data_parquet_path"])[['unique_id', 'ds']]
        # 3. Lấy dữ liệu feature lịch sử từ Feast
        feature_list = [
            "time_series_fv:y",
            # "time_series_fv:lag_5min",
            # "time_series_fv:lag_30min",
            # "time_series_fv:rolling_mean",
            # "time_series_fv:rolling_std",
            # "time_series_fv:lag_2h"
        ]

        df = store.get_historical_features(
            entity_df=entity_df[["unique_id", "ds"]],
            features=feature_list
        ).to_df()
        logger.info(f"Successfully fetched {len(df)} rows from Feast.")

        # 4. Split train/test
        logger.info("Splitting data into train and test sets...")
        split_ratio = config["data"]["split_ratio"]
        split_point = int(len(df) * split_ratio)
        train_df = df.iloc[:split_point]
        test_df = df.iloc[split_point:]
        logger.info(f"Train set size: {len(train_df)}, Test set size: {len(test_df)}")

        # 5. Train model
        logger.info("Training NBEATSx model...")
        model, n_params, score = train_patchtst(train_df, test_df, config)

        # 6. Log results to MLflow
        logger.info("Logging metrics and model to MLflow...")
        mlflow.log_params(config["model"]["nbeatsx"])
        mlflow.log_metric("n_parameters", n_params)
        mlflow.log_metric("MAPE", score)

        # Log model artifact (directory)
        mlflow.log_artifact("experiments/models/patchtst_model")
        # Log model in pytorch format for serving
        mlflow.pytorch.log_model(model.models[0], "model")

        

if __name__ == "__main__":
    run_pipeline()
