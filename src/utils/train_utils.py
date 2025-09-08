import pandas as pd
import json
import numpy as np
import logging
import os
import glob
def convert_speed_to_mbps(speed_str):
    value, unit = speed_str.split()
    value = float(value)
    unit = unit.lower()
    if unit == 'gbps':
        return value * 1000
    elif unit == 'mbps':
        return value
    else:
        return None

def process_traffic_data(df: pd.DataFrame, direction: str) -> pd.DataFrame:
    """
    Processes raw DataFrame to extract and format traffic data for NeuralForecast.
    
    Args:
        df (pd.DataFrame): The raw DataFrame from the Excel file.
        direction (str): The traffic direction, either 'in' or 'out'.
        
    Returns:
        pd.DataFrame: A formatted DataFrame with 'unique_id', 'ds', and 'y' columns.
    """
    if direction not in ['in', 'out']:
        raise ValueError("Direction must be 'in' or 'out'")
    
    col_name = f'KpiDataFindResult.traffic{direction.capitalize()}Str'
    
    processed_df = pd.DataFrame()
    processed_df['y'] = df[col_name].apply(convert_speed_to_mbps)
    processed_df['ds'] = pd.to_datetime(df['dataTime'], format='%d/%m/%Y %H:%M:%S')
    processed_df['unique_id'] = 'series_1'
    
    return processed_df

def mean_absolute_percentage_error(y_true, y_pred):
    """Calculates MAPE, avoiding division by zero."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero by replacing 0s in y_true with a small number or 1
    # Here we replace with 1, assuming traffic values are significantly larger.
    return np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))) * 100

def save_dict_to_json(data: dict, file_path: str):
    """Saves a dictionary to a JSON file."""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def load_and_process_data(
    folder_path: str,
    output_csv_path: str = None,
    traffic_direction: str = 'in',
    file_extension: str = 'xlsx',
    output_parquet_path: str = None,
    exog: bool = False
):
    """
    Tự động tìm, tải và xử lý tất cả các file trong một thư mục,
    sau đó gộp chúng lại và lưu thành một file CSV.

    Args:
        folder_path (str): Đường dẫn đến thư mục chứa các file cần xử lý.
        output_csv_path (str, optional): Đường dẫn để lưu file CSV kết quả. Nếu là None, DataFrame sẽ không được lưu.
        traffic_direction (str, optional): Hướng lưu lượng ('in' hoặc 'out'). Mặc định là 'in'.
        file_extension (str, optional): Phần mở rộng của các file cần tìm (không có dấu chấm). Mặc định là 'xlsx'.
    
    Returns:
        pd.DataFrame: DataFrame đã được gộp và xử lý, hoặc None nếu không có file nào được xử lý.
    """
    # 1. Tự động tìm tất cả các file trong thư mục
    logging.info(f"Searching for *.{file_extension} files in directory: '{folder_path}'")
    search_pattern = os.path.join(folder_path, f'*.{file_extension}')
    file_paths = glob.glob(search_pattern)

    if not file_paths:
        logging.warning("No matching files found in the directory. Exiting.")
        return None
    
    logging.info(f"Found {len(file_paths)} files. Starting processing...")
    
    list_of_dfs = []

    # 2. Lặp qua từng file và xử lý (logic tương tự như trước)
    for file_path in file_paths:
        try:
            logging.info(f"  - Processing file: {os.path.basename(file_path)}")
            df_raw = pd.read_excel(file_path, header=3)
            df_raw.columns = df_raw.columns.str.strip()
            df_processed = process_traffic_data(df_raw, direction=traffic_direction)
            df_processed.dropna(inplace=True)

            list_of_dfs.append(df_processed)
        except Exception as e:
            logging.error(f"    -> Error processing file {file_path}: {e}", exc_info=True)

    if not list_of_dfs:
        logging.warning("No files were processed successfully. Exiting.")
        return None

    # 3. Gộp và lưu kết quả
    logging.info("\nCombining data from all files...")
    combined_df = pd.concat(list_of_dfs, ignore_index=True)

    # Sort by timestamp and remove duplicates before calculating time-dependent features
    combined_df.sort_values('ds', inplace=True)
    combined_df.drop_duplicates(subset=['ds'], keep='last', inplace=True)
    if exog:
        combined_df['rolling_mean'] = combined_df['y'].rolling(window=7).mean()
        combined_df['rolling_std'] = combined_df['y'].rolling(window=7).std()
        combined_df['lag_5min'] = combined_df.groupby('unique_id')['y'].shift(1)      # 1 step back (5 min)
        combined_df['lag_30min'] = combined_df.groupby('unique_id')['y'].shift(6)
        combined_df['lag_2h'] = combined_df.groupby('unique_id')['y'].shift(24)
        # Drop rows with any NaNs, which are created by rolling/lag features
        combined_df.dropna(inplace=True)
    if output_csv_path:
        logging.info(f"Saving combined data to: '{output_csv_path}' and '{output_parquet_path}'")
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        combined_df.to_csv(output_csv_path, index=False)
        combined_df.to_parquet(output_parquet_path, index=False)
        logging.info(f"\nComplete! Data saved successfully. Total rows: {len(combined_df)}")
    
    return combined_df