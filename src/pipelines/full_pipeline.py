import argparse
import mlflow
import pandas as pd
import torch
import matplotlib.pyplot as plt
from src.models.train import train_model
from src.models.predict import perform_rolling_forecast
from src.models.evaluate import get_anomalies
from src.utils.config import load_config
from src.utils.logger import get_logger
from src.utils.metrics import get_metrics

logger = get_logger(__name__)

def run_pipeline(model_name: str):
    """
    Runs the full training, inference, and evaluation pipeline.

    Args:
        model_name: The name of the model to train (e.g., 'patchtst', 'timesnet').
    """
    config = load_config("config.yaml")
    mlflow.set_experiment(f"Full Pipeline - {model_name}")

    with mlflow.start_run(run_name=f"Full Pipeline Run - {model_name}") as parent_run:
        logger.info(f"Starting full pipeline for model: {model_name}")
        mlflow.log_param("model_name", model_name)
        mlflow.log_artifact("config.yaml")

        # --- Training ---
        logger.info("--- Starting Training ---")
        train_df = pd.read_parquet(config["data"]["processed_data_parquet_path"]).sort_values(['unique_id', 'ds'])
        nf, n_params = train_model(train_df=train_df, config=config, model_name=model_name.lower())
        mlflow.log_metric("n_parameters", n_params)
        logger.info("--- Training Finished ---")

        # --- Inference ---
        logger.info("--- Starting Inference (Rolling Forecast) ---")
        future_df = pd.read_parquet("data/processed/test_data.parquet")
        history_df = pd.read_parquet("data/processed/train_data.parquet")
        
        results = perform_rolling_forecast(
            nf=nf,
            history_df=history_df,
            future_df=future_df,
            silent=False
        )
        logger.info("--- Inference Finished ---")

        # --- Evaluation ---
        logger.info("--- Starting Evaluation ---")
        merged_df = pd.merge(future_df, results, on=['ds', 'unique_id'], how='inner')
        is_distribution_loss = any(f'{model_name}-median' in col for col in merged_df.columns)
        if merged_df.empty:
            logger.warning("No matching data between actuals and forecasts. Skipping evaluation.")
            mlflow.end_run()
            return

        output_csv = f"results/full_pipeline_prediction_{model_name.lower()}_{n_params/1e6:.2f}M.csv"
        merged_df.to_csv(output_csv, index=False)
        mlflow.log_artifact(output_csv)


        is_distribution_loss = any(f'{model_name}-median' in col for col in merged_df.columns)

        if is_distribution_loss:
            y_pred = torch.tensor(merged_df[f'{model_name}-median'].values, dtype=torch.float32)
        else:
            y_pred = torch.tensor(merged_df[model_name].values, dtype=torch.float32)

        y_true = torch.tensor(merged_df['y'].values, dtype=torch.float32)
        
        metrics = get_metrics(y_true, y_pred)
        mlflow.log_metrics(metrics)
        logger.info(f"Metrics: {metrics}")
        anomalies = get_anomalies(merged_df)
        logger.info(f"--- Detected {len(anomalies)} anomalies. ---")
        plt.plot(merged_df['ds'], merged_df['y'], label='Actual')
        if is_distribution_loss:
            plt.plot(merged_df['ds'], merged_df[f'{model_name}-median'], label='Forecast')
        else:
            plt.plot(merged_df['ds'], merged_df[model_name], label='Forecast')
        if not anomalies.empty:
            for anomaly in anomalies['ds']:
                plt.axvline(anomaly, color='red', linestyle='--', alpha=0.5)
        plt.legend()
        plt.title(f"{model_name} Forecast vs Actuals with Anomalies")
        plot_path = f"results/{model_name}_forecast_with_anomalies_{n_params/1e6:.2f}M.png"
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)


        logger.info("--- Evaluation Finished ---")
        
        logger.info("Pipeline finished successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full MLOps pipeline.")
    parser.add_argument("--model_name", type=str, choices=["PatchTST", "TimesNet", "NHITS"],required=True, help="Name of the model to train (e.g., patchtst, timesnet).")
    args = parser.parse_args()
    run_pipeline(args.model_name)