from src.utils.train_utils import load_and_process_data
from src.utils.config import load_config
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
from scipy.fftpack import fft
def prepare_training_data(
    raw_dir="data/raw",
    output_csv="data/processed/train_data.csv",
    traffic_direction="in",
    output_parquet_path=None,
    exog: bool = False
):
    df = load_and_process_data(
        folder_path=raw_dir,
        output_csv_path=output_csv,
        traffic_direction=traffic_direction,
        file_extension="xlsx",
        output_parquet_path=output_parquet_path,
        exog = exog
    )
    return df
def prepare_inference_data(
    raw_dir="data/raw/test",
    output_csv="data/processed/inference_data.csv",
    traffic_direction="in",
    output_parquet_path=None,
    exog: bool = False
):
    df = load_and_process_data(
        folder_path=raw_dir,
        output_csv_path=output_csv,
        traffic_direction=traffic_direction,
        file_extension="xlsx",
        output_parquet_path=output_parquet_path,
        exog = exog
    )


    return df
if __name__ == "__main__":
    config = load_config("config.yaml")
    df = prepare_training_data(
        raw_dir=config["data"]["path"],
        output_csv=config["data"]["processed_data_path"],
        traffic_direction="in",
        output_parquet_path=config["data"]["processed_data_parquet_path"],
        exog = False
    )
    # 1. Chuẩn bị dữ liệu
    # df = prepare_inference_data(output_parquet_path="data/processed/test_data.parquet", exog = False)
    print(df.head())