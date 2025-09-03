import time
import os
import pandas as pd
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from src.utils.config import load_config
from src.data.preprocess import prepare_training_data
from feast import FeatureStore
from feature_repo.feature_repo.feature_views import time_series_fv

# Load config
config = load_config("config.yaml")
WATCHED_DIR = config["data"]["path"]

# Setup logging
logging.basicConfig(
    filename=config["log"]["data_log"],
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

class ExcelFileHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(".xlsx"):
            logging.info(f"📂 New file detected: {event.src_path}")
            time.sleep(2)  # tránh file chưa ghi xong
            self.process_file(event.src_path)

    def process_file(self, filepath):
        try:
            logging.info("🔄 New file trigger → Reprocessing all data using prepare_training_data...")

            # 1. Reprocess toàn bộ dữ liệu
            processed_df = prepare_training_data(
                raw_dir=config["data"]["path"],
                output_csv=config["data"]["processed_data_path"],
                traffic_direction=config["data"]["traffic_direction"],
                output_parquet_path=config["data"]["processed_data_parquet_path"],
            )

            if processed_df is None or processed_df.empty:
                logging.warning("⚠️ Processed DataFrame is empty → Skipping Feast ingestion.")
                return

            # 2. Ghi dữ liệu vào online store
            logging.info(f"🚀 Writing {len(processed_df)} rows into Feast online store.")
            store = FeatureStore(repo_path=config["data"]["feature_store_path"])

            # Apply schema nếu cần (đảm bảo feature view được apply)
            store.apply([time_series_fv])

            # Ghi trực tiếp vào online store
            store.write_to_online_store(
                feature_view_name="time_series_fv",
                df=processed_df
            )

            logging.info("✅ Data successfully ingested into Feast online store.")

        except Exception as e:
            logging.error(f"❌ Error during file processing and Feast ingestion for {filepath}: {e}", exc_info=True)

def start_watching(folder_path):
    event_handler = ExcelFileHandler()
    observer = Observer()
    observer.schedule(event_handler, path=folder_path, recursive=False)
    observer.start()
    logging.info(f"👀 Started watching directory: {folder_path}")

    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        logging.info("🛑 Stopping directory watcher.")
        observer.stop()
    observer.join()

if __name__ == "__main__":
    start_watching(WATCHED_DIR)
