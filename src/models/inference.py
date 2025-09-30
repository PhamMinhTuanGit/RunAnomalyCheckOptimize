import mlflow
from mlflow import evaluate
from src.models.evaluate import perform_rolling_forecast, save_results
import pandas as pd
from neuralforecast import NeuralForecast
import matplotlib.pyplot as plt
import os
import torch
from src.utils.metrics import get_metrics
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import logging
logging.getLogger('statsforecast').setLevel(logging.ERROR)
logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)
logging.getLogger('mlflow').setLevel(logging.ERROR)
def plot_anomalies_interval_90(merged_path):
    merged_df = pd.read_csv(merged_path)
    merged_df['ds'] = pd.to_datetime(merged_df['ds'])
    if 'TimesNet-median' in merged_df.columns:
        save_dir = 'results/TimesNet_anomalies_plots_90'
    elif 'NHITS-median' in merged_df.columns:
        save_dir = 'results/NHITS_anomalies_plots_90'
    else:
        save_dir = 'results/PatchTST_anomalies_plots_90'
    # Vẽ dữ liệu ±1 ngày quanh mỗi anomaly
    
    
    lo_90_cols = [col for col in merged_df.columns if col.endswith('-lo-90')]
    hi_90_cols = [col for col in merged_df.columns if col.endswith('-hi-90')]
    anomalies = merged_df[
            (merged_df['y'] > merged_df[hi_90_cols[0]]) | 
            (merged_df['y'] < merged_df[lo_90_cols[0]])
        ]
    print(lo_90_cols, hi_90_cols)
    os.makedirs(save_dir, exist_ok=True)
    for idx, row in anomalies.iterrows():
        anomaly_time = row['ds']

        # Xác định khoảng thời gian xung quanh anomaly
        start_time = anomaly_time - timedelta(days=1)
        end_time = anomaly_time + timedelta(days=1)

        # Trích xuất dữ liệu trong khoảng đó
        window_data = merged_df[(merged_df['ds'] >= start_time) & (merged_df['ds'] <= end_time)]
        anomaly_window = merged_df[(merged_df['ds'] >= start_time) & (merged_df['ds'] <= end_time)]
        # Plot
        plt.figure(figsize=(10, 4))
        plt.plot(window_data['ds'], window_data['y'], label='Giá trị')
        plt.plot(window_data['ds'], window_data['PatchTST-median'], label='Dự đoán', color='orange')
        plt.fill_between(anomaly_window['ds'], anomaly_window[lo_90_cols[0]], anomaly_window[hi_90_cols[0]], alpha=0.3)
        plt.axvline(anomaly_time, color='red', linestyle='--', label='Anomaly')
        plt.title(f'Anomaly tại {anomaly_time}')
        plt.xlabel('Thời gian')
        plt.ylabel('Giá trị')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        plt.savefig(save_dir + f'/anomaly_{idx+1}_{anomaly_time.date()}.png')


def plot_anomalies_interval_95(merged_path):
    merged_df = pd.read_csv(merged_path)
    merged_df['ds'] = pd.to_datetime(merged_df['ds'])
    if 'TimesNet-median' in merged_df.columns:
        save_dir = 'results/TimesNet_anomalies_plots_95'
    elif 'NHITS-median' in merged_df.columns:
        save_dir = 'results/NHITS_anomalies_plots_95'
    else:
        save_dir = 'results/PatchTST_anomalies_plots_95'
    # Vẽ dữ liệu ±1 ngày quanh mỗi anomaly
    
    
    lo_95_cols = [col for col in merged_df.columns if col.endswith('-lo-95')]
    hi_95_cols = [col for col in merged_df.columns if col.endswith('-hi-95')]
    anomalies = merged_df[
            (merged_df['y'] > merged_df[hi_95_cols[0]]) | 
            (merged_df['y'] < merged_df[lo_95_cols[0]])
        ]
    print(lo_95_cols, hi_95_cols)
    os.makedirs(save_dir, exist_ok=True)
    for idx, row in anomalies.iterrows():
        anomaly_time = row['ds']

        # Xác định khoảng thời gian xung quanh anomaly
        start_time = anomaly_time - timedelta(days=1)
        end_time = anomaly_time + timedelta(days=1)

        # Trích xuất dữ liệu trong khoảng đó
        window_data = merged_df[(merged_df['ds'] >= start_time) & (merged_df['ds'] <= end_time)]
        anomaly_window = merged_df[(merged_df['ds'] >= start_time) & (merged_df['ds'] <= end_time)]
        # Plot
        plt.figure(figsize=(10, 4))
        plt.plot(window_data['ds'], window_data['y'], label='Giá trị')
        plt.plot(window_data['ds'], window_data['PatchTST-median'], label='Dự đoán', color='orange')
        plt.fill_between(anomaly_window['ds'], anomaly_window[lo_95_cols[0]], anomaly_window[hi_95_cols[0]], alpha=0.3)
        plt.axvline(anomaly_time, color='red', linestyle='--', label='Anomaly')
        plt.title(f'Anomaly tại {anomaly_time}')
        plt.xlabel('Thời gian')
        plt.ylabel('Giá trị')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        plt.savefig(save_dir + f'/anomaly_{idx+1}_{anomaly_time.date()}.png')


def plot_anomalies_interval_100(merged_path):
    merged_df = pd.read_csv(merged_path)
    merged_df['ds'] = pd.to_datetime(merged_df['ds'])
    if 'TimesNet-median' in merged_df.columns:
        save_dir = 'results/TimesNet_anomalies_plots_100'
    elif 'NHITS-median' in merged_df.columns:
        save_dir = 'results/NHITS_anomalies_plots_100'
    else:
        save_dir = 'results/PatchTST_anomalies_plots_100'
    # Vẽ dữ liệu ±1 ngày quanh mỗi anomaly
    
    
    lo_100_cols = [col for col in merged_df.columns if col.endswith('-lo-100')]
    hi_100_cols = [col for col in merged_df.columns if col.endswith('-hi-100')]
    anomalies = merged_df[
            (merged_df['y'] > merged_df[hi_100_cols[0]]) | 
            (merged_df['y'] < merged_df[lo_100_cols[0]])
        ]
    print(lo_100_cols, hi_100_cols)
    os.makedirs(save_dir, exist_ok=True)
    for idx, row in anomalies.iterrows():
        anomaly_time = row['ds']

        # Xác định khoảng thời gian xung quanh anomaly
        start_time = anomaly_time - timedelta(days=1)
        end_time = anomaly_time + timedelta(days=1)

        # Trích xuất dữ liệu trong khoảng đó
        window_data = merged_df[(merged_df['ds'] >= start_time) & (merged_df['ds'] <= end_time)]
        anomaly_window = merged_df[(merged_df['ds'] >= start_time) & (merged_df['ds'] <= end_time)]
        # Plot
        plt.figure(figsize=(10, 4))
        plt.plot(window_data['ds'], window_data['y'], label='Giá trị')
        plt.plot(window_data['ds'], window_data['PatchTST-median'], label='Dự đoán', color='orange')
        plt.fill_between(anomaly_window['ds'], anomaly_window[lo_100_cols[0]], anomaly_window[hi_100_cols[0]], alpha=0.3)
        plt.axvline(anomaly_time, color='red', linestyle='--', label='Anomaly')
        plt.title(f'Anomaly tại {anomaly_time}')
        plt.xlabel('Thời gian')
        plt.ylabel('Giá trị')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        plt.savefig(save_dir + f'/anomaly_{idx+1}_{anomaly_time.date()}.png')
def main():
    merged_path = 'results/prediction_nhits_7.71.csv'
    merged_df = pd.read_csv(merged_path)
    y_true = torch.tensor(merged_df['y'].values, dtype=torch.float32)
    y_pred = torch.tensor(merged_df['NHITS-median'].values, dtype=torch.float32)
    metrics = get_metrics(y_true, y_pred)
    # plot_anomalies_interval_90('patchtst_run_2_anomalies_817060.00.csv', merged_path)
    plot_anomalies_interval_90('results/prediction_nhits_7.71.csv', merged_path)
    print("Metrics:", metrics)
if __name__ == "__main__":
    main()