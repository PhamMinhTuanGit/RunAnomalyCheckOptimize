import pandas as pd
from src.utils.config import load_config
from src.data.preprocess import prepare_training_data
config = load_config("config.yaml")
df = prepare_training_data(
    raw_dir=config['data']['path'],
    output_csv=config["data"]["processed_data_path"],
    traffic_direction=config["data"]["traffic_direction"]
)
print(df.head())