from feast import FeatureStore
import mlflow
from src.models.train import train_nbeatsx
from src.utils.logger import get_logger
from src.utils.config import load_config
import pandas as pd
import logging
import os
import matplotlib.pyplot as plt 
from src.models.evaluate import perform_rolling_forecast
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MAPE, MAE, SMAPE, MSE
import torch

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

    if mlflow.active_run():
        mlflow.end_run()

    mlflow.set_experiment("N-BEATsx Experiment")
    with mlflow.start_run(run_name=config["experiment"]["run_name"], nested=True):
        mlflow.log_artifact("config.yaml")
        store = FeatureStore(repo_path=config['data']['feature_store_path'])  # sửa path repo Feast của bạn
        entity_df = pd.read_parquet(config["data"]["processed_data_parquet_path"])[['unique_id', 'ds']]
        # 3. Lấy dữ liệu feature lịch sử từ Feast
        feature_list = [
            "time_series_fv:y",
            "time_series_fv:lag_5min",
            "time_series_fv:lag_30min",
            "time_series_fv:rolling_mean",
            "time_series_fv:rolling_std",
            "time_series_fv:lag_2h"
        ]

        df = store.get_historical_features(
            entity_df=entity_df[["unique_id", "ds"]],
            features=feature_list
        ).to_df()
        logger.info(f"Successfully fetched {len(df)} rows from Feast.")

        # 4. Split train/test
        logger.info("Splitting data into train and test sets...")
        train_df = pd.read_parquet("data/processed/train_data.parquet")
        test_df = pd.read_parquet("data/processed/test_data.parquet")
        

        # 5. Train model
        logger.info("Training NBEATSx model...")
        model, n_params = train_nbeatsx(train_df, config)

        # 6. Log results to MLflow
        logger.info("Logging metrics and model to MLflow...")
        mlflow.log_params(config["model"]["nbeatsx"])
        mlflow.log_metric("n_parameters", n_params)


        # Log model artifact (directory)
        mlflow.log_artifact("experiments/models/nbeatsx_model")
        # Log model in pytorch format for serving
       

        # 7. Perform rolling forecast evaluation
        logger.info("Performing rolling forecast evaluation...")
        results = perform_rolling_forecast(
            future_df=test_df,
            history_df=train_df,
            nf=NeuralForecast.load("experiments/models/nbeatsx_model"),
            silent=True
        )
        
        if not results.empty:
            print(results.head())
            plt.figure(figsize=(12, 6))
            plt.plot(test_df['ds'], test_df['y'], label='Actual Future', color='green')
            plt.plot(results['ds'], results['NBEATSx-median'], label='Forecast', color='red', linestyle='--')
            plt.fill_between(results['ds'], results['NBEATSx-hi-90'], results['NBEATSx-lo-90'], color='red', alpha=0.2, label='Prediction Interval (90%)')
            plot_filename = f'{config["experiment"]["run_name"]}_nbeatsx_{n_params:.2f}.png'
            plt.savefig(plot_filename)
            mlflow.log_artifact(plot_filename)

            # Align lengths before calculating metrics
            merged_df = pd.merge(test_df, results, on=['ds', 'unique_id'], how='inner')
            y_true = torch.tensor(merged_df['y'].values, dtype=torch.float32)
            y_pred = torch.tensor(merged_df['NBEATSx-median'].values, dtype=torch.float32)
            y_history = torch.tensor(train_df['y'].values, dtype=torch.float32)

            mlflow.log_metric("rolling_forecast_mae", MAE()(y_pred, y_true))
            mlflow.log_metric("rolling_forecast_mse", MSE()(y_pred, y_true, y_insample=y_history))
            mlflow.log_metric("rolling_forecast_smape", MAPE()(y_pred, y_true, y_insample=y_history))
        else:
            logger.warning("Rolling forecast did not produce any results. Skipping plotting and metrics.")
        mlflow.end_run()

if __name__ == "__main__":
    run_pipeline()
