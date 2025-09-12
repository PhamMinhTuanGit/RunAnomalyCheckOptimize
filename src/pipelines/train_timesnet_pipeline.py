from feast import FeatureStore
import mlflow
from src.models.train import train_timesnet
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
    # Fix for NotImplementedError on MPS (Apple Silicon)
    # Sets a fallback to CPU for operations not supported on MPS.
    if torch.backends.mps.is_available():
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

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

    mlflow.set_experiment("TimesNet Experiment")
    with mlflow.start_run(run_name=config["experiment"]["run_name"], nested=True):
        mlflow.log_artifact("config.yaml")
        

        # 4. Split train/test
        logger.info("Splitting data into train and test sets...")
        train_df = pd.read_parquet("data/processed/train_data.parquet")
        test_df = pd.read_parquet("data/processed/test_data.parquet")
        

        # 5. Train model
        logger.info("Training NBEATSx model...")
        model, n_params = train_timesnet(train_df, config)

        # 6. Log results to MLflow
        logger.info("Logging metrics and model to MLflow...")
        mlflow.log_params(config["model"]["timesnet"])
        mlflow.log_metric("n_parameters", n_params)


        # Log model artifact (directory)
        mlflow.log_artifact("experiments/models/timesnet_model")
        # Log model in pytorch format for serving
       

        # 7. Perform rolling forecast evaluation
        logger.info("Performing rolling forecast evaluation...")
        results = perform_rolling_forecast(
            future_df=test_df,
            history_df=train_df,
            nf=NeuralForecast.load("experiments/models/timesnet_model"),
            silent=True
        )
        
        if not results.empty:
            # Hợp nhất kết quả dự báo và giá trị thực tế để dễ dàng so sánh và vẽ biểu đồ
            merged_df = pd.merge(test_df, results, on=['ds', 'unique_id'], how='inner')
            
            if merged_df.empty:
                logger.warning("Không có dữ liệu trùng khớp giữa giá trị thực tế và dự báo. Bỏ qua bước vẽ biểu đồ và tính toán metrics.")
            else:
                print(merged_df.head())

                # Xác định các điểm bất thường (actuals nằm ngoài khoảng dự báo 90%)
                anomalies = merged_df[
                    (merged_df['y'] > merged_df['TimesNet-hi-100']) | 
                    (merged_df['y'] < merged_df['TimesNet-lo-100'])
                ]
                logger.info(f"Phát hiện {len(anomalies)} điểm bất thường.")

            # Vẽ biểu đồ
            plt.figure(figsize=(18, 6))
            plt.plot(merged_df['ds'], merged_df['y'], label='Actual Future', color='green')
            plt.plot(merged_df['ds'], merged_df['TimesNet-median'], label='Forecast', color='red', linestyle='--')
            plt.fill_between(merged_df['ds'], merged_df['TimesNet-hi-100'], merged_df['TimesNet-lo-100'], color='red', alpha=0.2, label='Prediction Interval (90%)')
            # Đánh dấu các điểm bất thường bằng chấm đỏ
            plt.scatter(anomalies['ds'], anomalies['y'], color='red', s=50, zorder=5, label='Anomaly')
            plt.legend()
            plt.title("TimesNet Rolling Forecast with Anomalies")
            plt.savefig(f'{config["experiment"]["run_name"]}_{n_params:.2f}.png')
            mlflow.log_artifact(f'{config["experiment"]["run_name"]}_{n_params:.2f}.png')

            # Tính toán metrics sau khi đã căn chỉnh dữ liệu
            y_true = torch.tensor(merged_df['y'].values, dtype=torch.float32)
            y_pred = torch.tensor(merged_df['TimesNet-median'].values, dtype=torch.float32)
            y_pred = y_pred[:len(y_true)]
            y_history = torch.tensor(train_df['y'].values, dtype=torch.float32)
            mlflow.log_metric("rolling_forecast_mae", MAE()(y_pred, y_true))
            mlflow.log_metric("rolling_forecast_mse", MSE()(y_pred, y_true, y_insample=y_history))
            mlflow.log_metric("rolling_forecast_smape", MAPE()(y_pred, y_true, y_insample=y_history))
        else:
            logger.warning("Rolling forecast did not produce any results. Skipping plotting and metrics.")

        mlflow.end_run()

if __name__ == "__main__":
    run_pipeline()
