from src.utils.train_utils import load_and_process_data

def prepare_training_data(raw_dir="data/raw", output_csv="data/processed/train_data.csv", traffic_direction="in"):
    df = load_and_process_data(
        folder_path=raw_dir,
        output_csv_path=output_csv,
        traffic_direction=traffic_direction,
        file_extension="xlsx"
    )
    return df
