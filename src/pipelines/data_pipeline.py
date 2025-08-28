import time
import os
import pandas as pd
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from src.utils.config import load_config
from src.data.preprocess import prepare_training_data
from feast import FeatureStore
# Thư mục cần theo dõi
  # Thay bằng đường dẫn thực tế
config = load_config("config.yaml")
WATCHED_DIR = config['data']['path']
# Cấu hình logging
logging.basicConfig(
    filename=config['log']['data_log'],          # file lưu log
    level=logging.INFO,               # mức độ log: DEBUG, INFO, WARNING, ERROR, CRITICAL
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class ExcelFileHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(".xlsx"):
            logging.info(f"File mới phát hiện: {event.src_path}")
            self.process_file(event.src_path)

    def process_file(self, filepath):
        try:
            df = prepare_training_data(
                        raw_dir=config['data']['path'],
                        output_csv=config["data"]["processed_data_path"],
                        traffic_direction=config["data"]["traffic_direction"],
                        output_parquet_path=config["data"]["processed_data_parquet_path"]
                    )
            logging.info(f"Đọc thành công file: {os.path.basename(filepath)}")
            # Ví dụ: chỉ log 5 dòng đầu tiên dưới dạng chuỗi (nếu không quá dài)
            logging.info(f"Nội dung đầu file:\n{df.head().to_string()}")
            
            # TODO: xử lý dữ liệu
            self.save_to_feast(df)
        except Exception as e:
            logging.error(f"Lỗi khi xử lý file {filepath}: {e}")

    def save_to_feast(self, df):
        try:
            store = FeatureStore(repo_path=config['data']['feature_store_path'])
            # Ghi dữ liệu vào Feast
            store.apply([df])
            logging.info("Dữ liệu đã được ghi vào Feast thành công.")
        except Exception as e:
            logging.error(f"Lỗi khi ghi dữ liệu vào Feast: {e}")
def start_watching(folder_path):
    event_handler = ExcelFileHandler()
    observer = Observer()
    observer.schedule(event_handler, path=folder_path, recursive=False)
    observer.start()
    logging.info(f"Bắt đầu theo dõi thư mục: {folder_path}")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Dừng theo dõi thư mục.")
        observer.stop()
    observer.join()

if __name__ == "__main__":
    start_watching(WATCHED_DIR)
