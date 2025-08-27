from src.data.preprocess import prepare_training_data
from src.models.train import train_nhits
from src.utils.logger import get_logger
from src.utils.config import load_config

logger = get_logger("Pipeline")

def run_pipeline():
    config = load_config("config.yaml")

    # 1. Chuẩn bị dữ liệu
    df = prepare_training_data(
        raw_dir=config['data']['path'],
        output_csv="data/processed/train_data.csv",
        traffic_direction=config["data"]["traffic_direction"]
    )

    # 2. Split train/test
    train_size = int(len(df) * config["data"]["split_ratio"])
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]

    # 3. Train model
    model, score = train_nhits(train_df, test_df, config)
    logger.info(f"Pipeline completed. Final Test MAPE = {score:.2f}")

if __name__ == "__main__":
    run_pipeline()
