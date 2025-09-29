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

def plot_anomalies_interval_90(anomaly_path):
    anomaly_df = pd.read_csv(anomaly_path)

    # Chuyển cột thời gian về kiểu datetime
    anomaly_df['ds'] = pd.to_datetime(anomaly_df['ds'])

    # Sắp xếp theo thời gian nếu chưa được sắp xếp
    df = pd.read_parquet('./data/processed/test_data.parquet')

    df = df.sort_values('ds')
    if 'TimesNet-median' in anomaly_df.columns:
        save_dir = 'results/TimesNet_anomalies_plots'
    elif 'NHITS-median' in anomaly_df.columns:
        save_dir = 'results/NHITS_anomalies_plots'
    else:
        save_dir = 'results/PatchTST_anomalies_plots'
    # Vẽ dữ liệu ±1 ngày quanh mỗi anomaly
    merged_df = pd.read_csv('results/prediction_nhits_7.71.csv')
    merged_df['ds'] = pd.to_datetime(merged_df['ds'])

    lo_90_cols = [col for col in anomaly_df.columns if col.endswith('-lo-90')]
    hi_90_cols = [col for col in anomaly_df.columns if col.endswith('-hi-90')]
    print(lo_90_cols, hi_90_cols)
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
        plt.show()
        plt.savefig(save_dir + f'/anomaly_{idx+1}_{anomaly_time.date()}.png')
def save_results(forecasts_df: pd.DataFrame, future_df: pd.DataFrame, output_dir: str, model_name: str, silent: bool) -> str:
    """
    Merges forecasts with actuals, saves to a CSV, and returns the file path.
    """
    if not silent:
        print("Processing and saving results...")

    # Merge with actuals from the future_df for comparison
    results_df = pd.merge(future_df, forecasts_df, on=['unique_id', 'ds'], how='left')
    results_df.dropna(inplace=True)  # Drop any rows where prediction wasn't possible

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    

    # Save CSV
    forecast_csv_path = os.path.join(output_dir, f'rolling_forecast_{model_name}.csv')
    results_df.to_csv(forecast_csv_path, index=False)
    return forecast_csv_path


def patchtst_evaluate(model_path):
    nf = NeuralForecast.load(model_path)
    n_params = count_parameters(nf.models[0])
    mlflow.set_experiment(f"PatchTST Evaluation {n_params/1e6:.2f}M")
    with mlflow.start_run(run_name=config["experiment"]["run_name"], nested=True):
        mlflow.log_artifact("config.yaml")
        mlflow.log_params(config["model"]["patchtst"])
        mlflow.log_metric("n_parameters", n_params)
        # Log model artifact (directory)
        mlflow.log_artifact("experiments/models/patchtst_model")
        future_df = pd.read_parquet("data/processed/test_data.parquet")
        history_df = pd.read_parquet("data/processed/train_data.parquet")
        results = perform_rolling_forecast(
            future_df=future_df,
            history_df=history_df,
            nf=NeuralForecast.load("experiments/models/patchtst_model"),
            silent= True
        )
        
        if not results.empty:
            # Hợp nhất kết quả dự báo và giá trị thực tế để dễ dàng so sánh và vẽ biểu đồ
            merged_df = pd.merge(future_df, results, on=['ds', 'unique_id'], how='inner')
            merged_df.to_csv(f"results/prediction_patchtst_{n_params/1e6:.2f}.csv", index=False)
            mlflow.log_artifact(f"results/prediction_patchtst_{n_params/1e6:.2f}.csv")
            if merged_df.empty:
                print("Không có dữ liệu trùng khớp giữa giá trị thực tế và dự báo. Bỏ qua bước vẽ biểu đồ và tính toán metrics.")
            else:
                print(merged_df.head())

                # Xác định các điểm bất thường (actuals nằm ngoài khoảng dự báo 90%)
                anomalies = merged_df[
                    (merged_df['y'] > merged_df['PatchTST-hi-90']) | 
                    (merged_df['y'] < merged_df['PatchTST-lo-90'])
                ]
                print(f"Phát hiện {len(anomalies)} điểm bất thường.")
            filename = f'patchtst_{config["experiment"]["run_name"]}_anomalies_{n_params:.2f}.csv'
            anomalies.to_csv(filename, index=False)
            mlflow.log_metric("n_anomalies", len(anomalies))
            mlflow
            mlflow.log_artifact(filename)
            # Vẽ biểu đồ
            plt.figure(figsize=(18, 6))
            plt.plot(merged_df['ds'], merged_df['y'], label='Actual Future', color='green')
            plt.plot(merged_df['ds'], merged_df['PatchTST-median'], label='Forecast', color='red', linestyle='--')
            plt.fill_between(merged_df['ds'], merged_df['PatchTST-hi-90'], merged_df['PatchTST-lo-90'], color='red', alpha=0.2, label='Prediction Interval (90%)')
            # Đánh dấu các điểm bất thường bằng chấm đỏ
            plt.scatter(anomalies['ds'], anomalies['y'], color='red', s=50, zorder=5, label='Anomaly')
            plt.legend()
            plt.title("PatchTST Rolling Forecast with Anomalies")
            plt.savefig(f'{config["experiment"]["run_name"]}_{n_params:.2f}.png')
            mlflow.log_artifact(f'{config["experiment"]["run_name"]}_{n_params:.2f}.png')

            # Tính toán metrics sau khi đã căn chỉnh dữ liệu
            y_true = torch.tensor(merged_df['y'].values, dtype=torch.float32)
            y_pred = torch.tensor(merged_df['PatchTST-median'].values, dtype=torch.float32)
            metrics = get_metrics(y_true, y_pred)
            mlflow.log_metrics(metrics)  
            plot_anomalies_interval_90(filename)      
        else:  
            print("Rolling forecast did not produce any results. Skipping plotting and metrics.")
        mlflow.end_run()
def timesnet_evaluate(model_path):
    nf = NeuralForecast.load(model_path)
    n_params = count_parameters(nf.models[0])
    mlflow.set_experiment(f"TimesNet Evaluation {n_params:.2f}M")
    with mlflow.start_run(run_name=config["experiment"]["run_name"], nested=True):
        mlflow.log_artifact("config.yaml")
        mlflow.log_params(config["model"]["timesnet"])
        mlflow.log_metric("n_parameters", n_params)
        # Log model artifact (directory)
        mlflow.log_artifact(model_path)
        future_df = pd.read_parquet("data/processed/test_data.parquet")
        history_df = pd.read_parquet("data/processed/train_data.parquet")
        results = perform_rolling_forecast(
            future_df=future_df,
            history_df=history_df,
            nf=NeuralForecast.load(model_path),
            silent= True
        )
        
        if not results.empty:
            # Hợp nhất kết quả dự báo và giá trị thực tế để dễ dàng so sánh và vẽ biểu đồ
            merged_df = pd.merge(future_df, results, on=['ds', 'unique_id'], how='inner')
            merged_df.to_csv(f"results/prediction_timesnet_{n_params/1e6:.2f}.csv", index=False)
            mlflow.log_artifact(f"results/prediction_timesnet_{n_params/1e6:.2f}.csv")
            if merged_df.empty:
                print("Không có dữ liệu trùng khớp giữa giá trị thực tế và dự báo. Bỏ qua bước vẽ biểu đồ và tính toán metrics.")
            else:
                print(merged_df.head())

                # Xác định các điểm bất thường (actuals nằm ngoài khoảng dự báo 90%)
                anomalies = merged_df[
                    (merged_df['y'] > merged_df['TimesNet-hi-90']) | 
                    (merged_df['y'] < merged_df['TimesNet-lo-90'])
                ]
                print(f"Phát hiện {len(anomalies)} điểm bất thường.")
            filename = f'timesnet_{config["experiment"]["run_name"]}_anomalies_{n_params:.2f}.csv'
            anomalies.to_csv(filename, index=False)
            mlflow.log_metric("n_anomalies", len(anomalies))
            mlflow.log_artifact(filename, index=False)
            # Vẽ biểu đồ
            plt.figure(figsize=(18, 6))
            plt.plot(merged_df['ds'], merged_df['y'], label='Actual Future', color='green')
            plt.plot(merged_df['ds'], merged_df['TimesNet-median'], label='Forecast', color='red', linestyle='--')
            plt.fill_between(merged_df['ds'], merged_df['TimesNet-hi-90'], merged_df['TimesNet-lo-90'], color='red', alpha=0.2, label='Prediction Interval (90%)')
            # Đánh dấu các điểm bất thường bằng chấm đỏ
            plt.scatter(anomalies['ds'], anomalies['y'], color='red', s=50, zorder=5, label='Anomaly')
            plt.legend()
            plt.title("TimesNet Rolling Forecast with Anomalies")
            plt.savefig(f'{config["experiment"]["run_name"]}_{n_params:.2f}.png')
            mlflow.log_artifact(f'{config["experiment"]["run_name"]}_{n_params:.2f}.png')
            # Tính toán metrics sau khi đã căn chỉnh dữ liệu
            y_true = torch.tensor(merged_df['y'].values, dtype=torch.float32)
            y_pred = torch.tensor(merged_df['TimesNet-median'].values, dtype=torch.float32)
            metrics = get_metrics(y_true, y_pred)
            mlflow.log_metrics(metrics) 
            plot_anomalies_interval_90(filename)      
        else:  
            print("Rolling forecast did not produce any results. Skipping plotting and metrics.")
        mlflow.end_run()
def nhits_evaluate(model_path):
    nf = NeuralForecast.load(model_path)
    n_params = count_parameters(nf.models[0])
    mlflow.set_experiment(f"NHITS Evaluation {n_params:.2f}M")
    with mlflow.start_run(run_name=config["experiment"]["run_name"], nested=True):
        mlflow.log_artifact("config.yaml")
        mlflow.log_params(config["model"]["nhits"])
        mlflow.log_metric("n_parameters", n_params)
        # Log model artifact (directory)
        mlflow.log_artifact("experiments/models/nhits_model")
        future_df = pd.read_parquet("data/processed/test_data.parquet")
        history_df = pd.read_parquet("data/processed/train_data.parquet")
        results = perform_rolling_forecast(
            future_df=future_df,
            history_df=history_df,
            nf=NeuralForecast.load("experiments/models/nhits_model"),
            silent= True
        )
        
        if not results.empty:
            # Hợp nhất kết quả dự báo và giá trị thực tế để dễ dàng so sánh và vẽ biểu đồ
            merged_df = pd.merge(future_df, results, on=['ds', 'unique_id'], how='inner')
            merged_df.to_csv(f"results/prediction_nhits_{n_params/1e6:.2f}.csv", index=False)
            mlflow.log_artifact(f"results/prediction_nhits_{n_params/1e6:.2f}.csv")
            if merged_df.empty:
                print("Không có dữ liệu trùng khớp giữa giá trị thực tế và dự báo. Bỏ qua bước vẽ biểu đồ và tính toán metrics.")
            else:
                print(merged_df.head())

                # Xác định các điểm bất thường (actuals nằm ngoài khoảng dự báo 90%)
                anomalies = merged_df[
                    (merged_df['y'] > merged_df['NHITS-hi-90']) | 
                    (merged_df['y'] < merged_df['NHITS-lo-90'])
                ]
                print(f"Phát hiện {len(anomalies)} điểm bất thường.")
            filename = f'nhits_{config["experiment"]["run_name"]}_anomalies_{n_params:.2f}.csv'
            anomalies.to_csv(filename, index=False)
            mlflow.log_metric("n_anomalies", len(anomalies))
            mlflow.log_artifact(filename)
            # Vẽ biểu đồ
            plt.figure(figsize=(18, 6))
            plt.plot(merged_df['ds'], merged_df['y'], label='Actual Future', color='green')
            plt.plot(merged_df['ds'], merged_df['NHITS-median'], label='Forecast', color='red', linestyle='--')
            plt.fill_between(merged_df['ds'], merged_df['NHITS-hi-90'], merged_df['NHITS-lo-90'], color='red', alpha=0.2, label='Prediction Interval (90%)')
            # Đánh dấu các điểm bất thường bằng chấm đỏ
            plt.scatter(anomalies['ds'], anomalies['y'], color='red', s=50, zorder=5, label='Anomaly')
            plt.legend()
            plt.title("NHITS Rolling Forecast with Anomalies")
            plt.savefig(f'{config["experiment"]["run_name"]}_{n_params:.2f}.png')
            mlflow.log_artifact(f'{config["experiment"]["run_name"]}_{n_params:.2f}.png')
            # Tính toán metrics sau khi đã căn chỉnh dữ liệu
            y_true = torch.tensor(merged_df['y'].values, dtype=torch.float32)
            y_pred = torch.tensor(merged_df['NHITS-median'].values, dtype=torch.float32)
            metrics = get_metrics(y_true, y_pred)
            mlflow.log_metrics("Evaluation Metrics",metrics) 
            plot_anomalies_interval_90(filename)       
        else:  
            print("Rolling forecast did not produce any results. Skipping plotting and metrics.")
        mlflow.end_run()


def timesnet_evaluate_non_distribution(model_path):
    nf = NeuralForecast.load(model_path)
    n_params = count_parameters(nf.models[0])

    mlflow.set_experiment(f"TimesNet Evaluation {n_params:.2f}M")

    with mlflow.start_run(run_name=config["experiment"]["run_name"], nested=True):
        mlflow.log_artifact("config.yaml")
        mlflow.log_params(config["model"]["timesnet"])
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

        # Merge với dữ liệu thực tế để so sánh
        merged_df = pd.merge(future_df, results, on=['ds', 'unique_id'], how='inner')

        if merged_df.empty:
            print("⚠️ Không có dữ liệu trùng khớp giữa thực tế và dự báo.")
            mlflow.end_run()
            return

        output_csv = f"results/prediction_timesnet_{n_params/1e6:.2f}.csv"
        merged_df.to_csv(output_csv, index=False)
        mlflow.log_artifact(output_csv)

        print(merged_df.head())

        # Vẽ biểu đồ không có khoảng dự báo
        plt.figure(figsize=(18, 6))
        plt.plot(merged_df['ds'], merged_df['y'], label='Actual Future', color='green')
        plt.plot(merged_df['ds'], merged_df['TimesNet'], label='Forecast', color='red', linestyle='--')
        plt.title("TimesNet Rolling Forecast")
        plt.legend()
        fig_path = f'{config["experiment"]["run_name"]}_{n_params:.2f}.png'
        plt.savefig(fig_path)
        mlflow.log_artifact(fig_path)

        # Tính metrics
        y_true = torch.tensor(merged_df['y'].values, dtype=torch.float32)
        y_pred = torch.tensor(merged_df['TimesNet-median'].values, dtype=torch.float32)
        metrics = get_metrics(y_true, y_pred)
        mlflow.log_metrics(metrics)

        mlflow.end_run()

if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    model_path = args.model_path
    # patchtst_evaluate(model_path)
    timesnet_evaluate_non_distribution(model_path)