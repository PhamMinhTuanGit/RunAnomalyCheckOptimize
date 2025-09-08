import os
import time
import pandas as pd
from feast import FeatureStore
from datetime import datetime
from src.utils.config import load_config
from src.utils.logger import get_logger
from src.models.predict import forecast
from src.data.preprocess import prepare_inference_data
from src.models.evaluate import perform_rolling_forecast, save_results

def real_time_forecast_loop(
    future_df: pd.DataFrame,
    history_df: pd.DataFrame,
    feast_repo_path: str,
    model_checkpoint_path: str,
    eval_output_path: str = "results/eval_results.csv",
    sleep_time: int = 5,
    H: int = 12
):
    from feast import FeatureStore
    from neuralforecast import NeuralForecast
    from mlflow.models import evaluate
    import pandas as pd
    import os
    import time

    print("[+] Loading model from checkpoint...")
    nf = NeuralForecast.load(model_checkpoint_path)
    model = nf.models[0]

    print("[+] Loading Feast feature store...")
    store = FeatureStore(repo_path=feast_repo_path)

    # Ensure output folder exists
    os.makedirs(os.path.dirname(eval_output_path), exist_ok=True)

    full_future_df = future_df.copy()
    full_future_df = full_future_df.sort_values("ds").reset_index(drop=True)

    while len(full_future_df) >= H:
        current_hist = history_df.iloc[H:].copy()
        forecast_start = full_future_df.iloc[0]["ds"]

        print(f"[✓] Forecasting from: {forecast_start} for {H} steps")

        # Push current H-step history to Feast
        for i in range(H):
            store.push("time_series_push_source", current_hist.iloc[[i]])

        # Update model's internal data


        # Forecast H steps
        forecast_df = nf.predict().sort_values("ds").reset_index(drop=True)

        # Get ground truth to compare
        ground_truth_df = full_future_df.iloc[:H][["unique_id", "ds", "y"]]
        print(ground_truth_df.head())
        # Merge and evaluate
        eval_df = forecast_df.merge(ground_truth_df, on=["unique_id", "ds"])
        print(eval_df.head())

        
        
        print(f"[🎯] MAPE (H={H}) at {forecast_start}: {mape:.4f}")

        # Log to file
        with open(eval_output_path, "a") as f:
            f.write(f"{forecast_start},{mape}\n")

        # Xoá H dòng đã dùng
        full_future_df = full_future_df.iloc[H:].reset_index(drop=True)

        time.sleep(sleep_time)

if __name__ == "__main__":
    future_df = pd.read_parquet("data/processed/test_data.parquet")
    history_df = pd.read_parquet("data/processed/train_data.parquet")
    real_time_forecast_loop(
        future_df=future_df,
        history_df=history_df,
        feast_repo_path="feature_repo/feature_repo",
        model_checkpoint_path="experiments/models/patchtst_model",
    )