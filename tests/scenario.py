'''
Ở scenario này, interval giữa mỗi lần nhận dữ liệu là 5 giây để phục vụ mục đích kiểm thử.

'''


import pandas as pd
import time
from datetime import datetime
from src.utils.train_utils import *
def iterate_excel_file_every_5seconds(file_path):
    xls = pd.read_excel(file_path, header=3)
    # convert datetime trước khi yield
    xls['dataTime'] = pd.to_datetime(xls['dataTime'], format='%d/%m/%Y %H:%M:%S', utc = True)
    for _, row in xls.iterrows():
        row_df = row.to_frame().T
        
        yield row_df
        time.sleep(5)

def transform_row(row):
    df_processed = process_traffic_data(row, direction="in")
    df_processed.dropna(inplace=True)
    return df_processed
 

output_csv = "predictions_log.csv"
# Tạo file mới với header
pd.DataFrame(columns=["timestamp","entity_id","prediction"]).to_csv(output_csv, index=False)
