from neuralforecast import NeuralForecast
from tqdm import tqdm
import pandas as pd
import mlflow
import matplotlib.pyplot as plt
import os
import torch
import argparse
from src.models.train import count_parameters
from src.utils.logger import get_logger
from src.utils.config import load_config
from src.utils.metrics import get_metrics
from datetime import timedelta
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
config = load_config("config.yaml")

def setup_parser()->argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained model using rolling forecast.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved model directory.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to evaluate (e.g., PatchTST, TimesNet, NHITS).")
    
    return parser

def perform_rolling_forecast(nf: NeuralForecast, history_df: pd.DataFrame, future_df: pd.DataFrame, silent: bool) -> pd.DataFrame:
    """
    Performs a rolling forecast simulation.

    Args:
        nf: The initialized NeuralForecast object.
        history_df: The initial DataFrame to start forecasting from.
        future_df: The DataFrame with future actual values to iterate over.
        silent: Flag to disable progress bar.

    Returns:
        A DataFrame containing all concatenated forecasts.
    """
    h = nf.models[0].h  # Get forecast horizon from the loaded model
    all_forecasts = []

    if not silent:
        print(f"Performing rolling forecast with horizon h={h}...")

    # Use tqdm for a progress bar, which is disabled in silent mode
    for i in tqdm(range(0, len(future_df), h), desc="Rolling Forecast Steps", disable=silent):
        # Predict h steps into the future from the end of the current history
        forecast = nf.predict(df=history_df)
        all_forecasts.append(forecast)

        # Update history by appending the *actual* observed data for the next iteration
        actuals_for_step = future_df.iloc[i: i + h]
        if actuals_for_step.empty:
            break
        history_df = pd.concat([history_df, actuals_for_step])
        print(f'Progress: {i}/{len(future_df)}')
    return pd.concat(all_forecasts).reset_index()

def plot_anomalies_interval_90(anomaly_path, model_name):
    anomaly_df = pd.read_csv(anomaly_path)

    # Chuyển cột thời gian về kiểu datetime
    anomaly_df['ds'] = pd.to_datetime(anomaly_df['ds'])

    # Sắp xếp theo thời gian nếu chưa được sắp xếp
    df = pd.read_parquet('./data/processed/test_data.parquet')

    df = df.sort_values('ds')
    save_dir = f'results/{model_name}_anomalies_plots'
    os.makedirs(save_dir, exist_ok=True)
    
    # Vẽ dữ liệu ±1 ngày quanh mỗi anomaly
    merged_df = pd.read_csv(anomaly_path.replace('anomalies', 'prediction'))
    merged_df['ds'] = pd.to_datetime(merged_df['ds'])

    lo_90_cols = [col for col in merged_df.columns if col.endswith('-lo-90')]
    hi_90_cols = [col for col in merged_df.columns if col.endswith('-hi-90')]
    
    for idx, row in anomaly_df.iterrows():
        anomaly_time = row['ds']

        # Xác định khoảng thời gian xung quanh anomaly
        start_time = anomaly_time - timedelta(days=1)
        end_time = anomaly_time + timedelta(days=1)

        # Trích xuất dữ liệu trong khoảng đó
        window_data = df[(df['ds'] >= start_time) & (df['ds'] <= end_time)]
        anomaly_window = merged_df[(merged_df['ds'] >= start_time) & (merged_df['ds'] <= end_time)]
        # Plot
        plt.figure(figsize=(10, 4))
        plt.plot(window_data['ds'], window_data['y'], label='Giá trị')
        plt.fill_between(anomaly_window['ds'], anomaly_window[lo_90_cols[0]], anomaly_window[hi_90_cols[0]], alpha=0.3)
        plt.axvline(anomaly_time, color='red', linestyle='--', label='Anomaly')
        plt.title(f'Anomaly tại {anomaly_time}')
        plt.xlabel('Thời gian')
        plt.ylabel('Giá trị')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'anomaly_{idx+1}_{anomaly_time.date()}.png'))
        plt.close()

def plot_anomalies_interval_95(anomaly_path, model_name):
    anomaly_df = pd.read_csv(anomaly_path)

    # Chuyển cột thời gian về kiểu datetime
    anomaly_df['ds'] = pd.to_datetime(anomaly_df['ds'])

    # Sắp xếp theo thời gian nếu chưa được sắp xếp
    df = pd.read_parquet('./data/processed/test_data.parquet')

    df = df.sort_values('ds')
    save_dir = f'results/{model_name}_anomalies_plots'
    os.makedirs(save_dir, exist_ok=True)
    
    # Vẽ dữ liệu ±1 ngày quanh mỗi anomaly
    merged_df = pd.read_csv(anomaly_path.replace('anomalies', 'prediction'))
    merged_df['ds'] = pd.to_datetime(merged_df['ds'])

    lo_95_cols = [col for col in merged_df.columns if col.endswith('-lo-95')]
    hi_95_cols = [col for col in merged_df.columns if col.endswith('-hi-95')]
    
    if not lo_95_cols or not hi_95_cols:
        print("95% prediction interval columns not found.")
        return

    for idx, row in anomaly_df.iterrows():
        anomaly_time = row['ds']

        # Xác định khoảng thời gian xung quanh anomaly
        start_time = anomaly_time - timedelta(days=1)
        end_time = anomaly_time + timedelta(days=1)

        # Trích xuất dữ liệu trong khoảng đó
        window_data = df[(df['ds'] >= start_time) & (df['ds'] <= end_time)]
        anomaly_window = merged_df[(merged_df['ds'] >= start_time) & (merged_df['ds'] <= end_time)]
        # Plot
        plt.figure(figsize=(10, 4))
        plt.plot(window_data['ds'], window_data['y'], label='Giá trị')
        plt.fill_between(anomaly_window['ds'], anomaly_window[lo_95_cols[0]], anomaly_window[hi_95_cols[0]], alpha=0.3)
        plt.axvline(anomaly_time, color='red', linestyle='--', label='Anomaly')
        plt.title(f'Anomaly tại {anomaly_time} (95% interval)')
        plt.xlabel('Thời gian')
        plt.ylabel('Giá trị')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'anomaly_{idx+1}_{anomaly_time.date()}_95.png'))
        plt.close()

def plot_anomalies_interval_100(anomaly_path, model_name):
    anomaly_df = pd.read_csv(anomaly_path)

    # Chuyển cột thời gian về kiểu datetime
    anomaly_df['ds'] = pd.to_datetime(anomaly_df['ds'])

    # Sắp xếp theo thời gian nếu chưa được sắp xếp
    df = pd.read_parquet('./data/processed/test_data.parquet')

    df = df.sort_values('ds')
    save_dir = f'results/{model_name}_anomalies_plots'
    os.makedirs(save_dir, exist_ok=True)
    
    # Vẽ dữ liệu ±1 ngày quanh mỗi anomaly
    merged_df = pd.read_csv(anomaly_path.replace('anomalies', 'prediction'))
    merged_df['ds'] = pd.to_datetime(merged_df['ds'])

    lo_100_cols = [col for col in merged_df.columns if col.endswith('-lo-100')]
    hi_100_cols = [col for col in merged_df.columns if col.endswith('-hi-100')]

    if not lo_100_cols or not hi_100_cols:
        print("100% prediction interval columns not found.")
        return
    
    for idx, row in anomaly_df.iterrows():
        anomaly_time = row['ds']

        # Xác định khoảng thời gian xung quanh anomaly
        start_time = anomaly_time - timedelta(days=1)
        end_time = anomaly_time + timedelta(days=1)

        # Trích xuất dữ liệu trong khoảng đó
        window_data = df[(df['ds'] >= start_time) & (df['ds'] <= end_time)]
        anomaly_window = merged_df[(merged_df['ds'] >= start_time) & (merged_df['ds'] <= end_time)]
        # Plot
        plt.figure(figsize=(10, 4))
        plt.plot(window_data['ds'], window_data['y'], label='Giá trị')
        plt.fill_between(anomaly_window['ds'], anomaly_window[lo_100_cols[0]], anomaly_window[hi_100_cols[0]], alpha=0.3)
        plt.axvline(anomaly_time, color='red', linestyle='--', label='Anomaly')
        plt.title(f'Anomaly tại {anomaly_time} (100% interval)')
        plt.xlabel('Thời gian')
        plt.ylabel('Giá trị')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'anomaly_{idx+1}_{anomaly_time.date()}_100.png'))
        plt.close()


def evaluate(model_path, model_name):
    nf = NeuralForecast.load(model_path)
    n_params = count_parameters(nf.models[0])
    
    mlflow.set_experiment(f"{model_name} Evaluation {n_params/1e6:.2f}M")
    with mlflow.start_run(run_name=config["experiment"]["run_name"], nested=True):
        mlflow.log_artifact("config.yaml")
        mlflow.log_params(config["model"][model_name.lower()])
        mlflow.log_metric("n_parameters", n_params)
        mlflow.log_artifact(model_path)

        future_df = pd.read_parquet("data/processed/test_data.parquet")
        history_df = pd.read_parquet("data/processed/train_data.parquet")

        results = perform_rolling_forecast(
            future_df=future_df,
            history_df=history_df,
            nf=nf,
            silent=True
        )

        if results.empty:
            print("⚠️ Rolling forecast did not produce any results. Skipping plotting and metrics.")
            mlflow.end_run()
            return

        merged_df = pd.merge(future_df, results, on=['ds', 'unique_id'], how='inner')

        if merged_df.empty:
            print("⚠️ No matching data between actuals and forecasts.")
            mlflow.end_run()
            return

        output_csv = f"results/prediction_{model_name.lower()}_{n_params/1e6:.2f}.csv"
        merged_df.to_csv(output_csv, index=False)
        mlflow.log_artifact(output_csv)

        print(merged_df.head())

        # Check if the model was trained with DistributionLoss
        is_distribution_loss = any(f'{model_name}-median' in col for col in merged_df.columns)

        plt.figure(figsize=(18, 6))
        plt.plot(merged_df['ds'], merged_df['y'], label='Actual Future', color='green')

        if is_distribution_loss:
            plt.plot(merged_df['ds'], merged_df[f'{model_name}-median'], label='Forecast', color='red', linestyle='--')
            plt.fill_between(merged_df['ds'], merged_df[f'{model_name}-lo-90'], merged_df[f'{model_name}-hi-90'], color='red', alpha=0.2, label='Prediction Interval (90%)')
            
            anomalies = merged_df[
                (merged_df['y'] > merged_df[f'{model_name}-hi-90']) | 
                (merged_df['y'] < merged_df[f'{model_name}-lo-90'])
            ]
            print(f"Found {len(anomalies)} anomalies.")
            
            plt.scatter(anomalies['ds'], anomalies['y'], color='red', s=50, zorder=5, label='Anomaly')
            
            anomaly_filename = f'results/{model_name.lower()}_{config["experiment"]["run_name"]}_anomalies_{n_params:.2f}.csv'
            anomalies.to_csv(anomaly_filename, index=False)
            mlflow.log_metric("n_anomalies", len(anomalies))
            mlflow.log_artifact(anomaly_filename)
            
            y_pred = torch.tensor(merged_df[f'{model_name}-median'].values, dtype=torch.float32)
            
            plot_anomalies_interval_90(anomaly_filename, model_name)

        else: # Non-distributional model
            plt.plot(merged_df['ds'], merged_df[model_name], label='Forecast', color='red', linestyle='--')
            y_pred = torch.tensor(merged_df[model_name].values, dtype=torch.float32)

        plt.legend()
        plt.title(f"{model_name} Rolling Forecast with Anomalies")
        fig_path = f'results/{config["experiment"]["run_name"]}_{n_params:.2f}.png'
        plt.savefig(fig_path)
        mlflow.log_artifact(fig_path)

        y_true = torch.tensor(merged_df['y'].values, dtype=torch.float32)
        metrics = get_metrics(y_true, y_pred)
        mlflow.log_metrics(metrics)
        
        mlflow.end_run()

def get_anomalies(df):
    if 'TimesNet-median' in df.columns:
        model_name = 'TimesNet'
    elif 'NHITS-median' in df.columns:
        model_name = 'NHITS'
    elif 'PatchTST-median' in df.columns:
        model_name = 'PatchTST'
    else:
        print("No recognized model prediction columns found.")
        return pd.DataFrame()

    anomalies = df[
        (df['y'] > df[f'{model_name}-hi-100']) | 
        (df['y'] < df[f'{model_name}-lo-100'])
    ]
    print(f"Found {len(anomalies)} anomalies.")
    return anomalies

if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    evaluate(args.model_path, args.model_name)
