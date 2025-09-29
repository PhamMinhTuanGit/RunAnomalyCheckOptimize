import mlflow
import pandas as pd

# Nếu dùng server thì set URI, nếu local thì có thể bỏ qua
# mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Lấy toàn bộ experiment
experiments = [exp.experiment_id for exp in mlflow.get_experiment()]

# Tìm tất cả runs trong các experiment
df = mlflow.search_runs(experiment_ids=experiments)

# Xuất ra CSV
df.to_csv("all_mlflow_experiments.csv", index=False)

print(f"Đã xuất {len(df)} runs từ {len(experiments)} experiment vào all_mlflow_experiments.csv")
