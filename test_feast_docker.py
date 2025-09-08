import mlflow
from mlflow import evaluate
from src.models.evaluate import perform_rolling_forecast, save_results
import pandas as pd
from neuralforecast import NeuralForecast
import matplotlib.pyplot as plt
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import logging
logging.getLogger('statsforecast').setLevel(logging.ERROR)
logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)
logging.getLogger('mlflow').setLevel(logging.ERROR)

def main():
    future_df = pd.read_parquet("data/processed/test_data.parquet")
    history_df = pd.read_parquet("data/processed/train_data.parquet")
    results = perform_rolling_forecast(
        future_df=future_df,
        history_df=history_df,
        nf=NeuralForecast.load("experiments/models/patchtst_model"),
        silent= True
    )
    print(results.head())
    plt.figure(figsize=(12, 6))
    plt.plot(history_df['ds'], history_df['y'], label='History', color='blue')
    plt.plot(future_df['ds'], future_df['y'], label='Actual Future', color='green')
    plt.plot(results['ds'], results['PatchTST-median'], label='Forecast', color='red', linestyle='--')
    plt.fill_between(results['ds'], results['PatchTST-hi-90'], results['PatchTST-lo-90'], color='red', alpha=0.2, label='Prediction Interval (10%-90%)')
    plt.savefig('rolling_forecast.png')
if __name__ == "__main__":
    main()