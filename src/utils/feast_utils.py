import pandas as pd
from feast import FeatureStore

def get_historical_features_for_neuralforecast(entity_df: pd.DataFrame, feature_repo_path: str) -> pd.DataFrame:
    """
    Connects to the feature store, retrieves historical features, and formats
    them into the structure required by NeuralForecast.

    Args:
        entity_df: A pandas DataFrame containing the entity keys and timestamps.
                   It must contain 'unique_id' and 'ds' columns.
        feature_repo_path: Path to the feature repository.

    Returns:
        A pandas DataFrame with historical features, ready for NeuralForecast.
    """
    # Feast expects the timestamp column to be named 'event_timestamp'
    if "ds" not in entity_df.columns:
        raise ValueError("Input DataFrame must contain a 'ds' column.")
    
    entity_df_for_feast = entity_df.rename(columns={"ds": "event_timestamp"})

    # Ensure the timestamp column is of datetime type and is timezone-aware (UTC)
    entity_df_for_feast["event_timestamp"] = pd.to_datetime(entity_df_for_feast["event_timestamp"])
    if entity_df_for_feast["event_timestamp"].dt.tz is None:
        entity_df_for_feast["event_timestamp"] = entity_df_for_feast["event_timestamp"].dt.tz_localize('UTC')
    else:
        entity_df_for_feast["event_timestamp"] = entity_df_for_feast["event_timestamp"].dt.tz_convert('UTC')

    store = FeatureStore(repo_path=feature_repo_path)

    # Retrieve historical features
    training_df = store.get_historical_features(
        entity_df=entity_df_for_feast,
        features=[
            "interface_traffic:y",
            "transformed_interface_traffic:lag_1h",
        ],
    ).to_df()

    # Rename the timestamp column back to 'ds' for NeuralForecast
    training_df = training_df.rename(columns={"event_timestamp": "ds"})

    return training_df
