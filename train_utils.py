import mlflow
import pandas as pd

# Nếu dùng server thì set URI, nếu local thì có thể bỏ qua
# mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Lấy toàn bộ experiment
experiments = mlflow.search_experiments()
experiment_ids = [exp.experiment_id for exp in experiments]
experiment_name_map = {exp.experiment_id: exp.name for exp in experiments}

# Tìm tất cả runs trong các experiment
df = mlflow.search_runs(experiment_ids=experiment_ids)

# Thêm cột tên experiment nếu dataframe không rỗng
if not df.empty:
    df['experiment_name'] = df['experiment_id'].map(experiment_name_map)
    # Lọc những run đã hoàn thành (status 'FINISHED') và có metric đánh giá
    df = df[df['status'] == 'FINISHED'].copy()
    df.dropna(subset=['metrics.rolling_forecast_mae'], inplace=True)

# Định nghĩa các cột quan trọng cần trích xuất
important_columns = [
    'experiment_name',
    'tags.mlflow.runName',
    'metrics.n_parameters',
    'metrics.rolling_forecast_mae',
    'metrics.rolling_forecast_mse',
    'metrics.rolling_forecast_smape',
    
    'params.max_steps',
    'params.lr',
    
]

# Lọc ra những cột quan trọng thực sự tồn tại trong DataFrame
existing_important_columns = [col for col in important_columns if col in df.columns]


# Xuất ra file
if not df.empty:
    # Lấy các cột cần thiết và tạo một bản sao để tránh SettingWithCopyWarning
    df_summary = df[existing_important_columns].copy()

    # Đổi tên cột để dễ đọc hơn
    df_summary.rename(columns={
        'tags.mlflow.runName': 'run_name',
        'params.lr': 'lr',
        'params.max_steps': 'max_steps',
        'metrics.n_parameters': 'n_parameters',
        'metrics.rolling_forecast_mae': 'MAE',
        'metrics.rolling_forecast_mse': 'MSE',
        'metrics.rolling_forecast_smape': 'SMAPE'
    }, inplace=True)

    df_summary.to_excel("mlflow_runs_summary.xlsx", index=False)
    print(f"Đã xuất tóm tắt {len(df)} runs hoàn chỉnh từ {len(experiments)} experiment vào mlflow_runs_summary.xlsx")
else:
    print("Không tìm thấy run nào hoàn chỉnh để xuất ra file.")
