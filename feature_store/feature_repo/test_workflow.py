import subprocess
from datetime import datetime
import os
import pandas as pd

from feast import FeatureStore


def run_demo():
    repo_path = os.path.abspath(os.path.dirname(__file__))
    store = FeatureStore(repo_path=repo_path)
    print("\n--- Run feast apply ---")
    subprocess.run(["feast", "apply"], cwd=repo_path)

    print("\n--- Historical features for training ---")
    fetch_historical_features_entity_df(store)

    # print("\n--- Load features into online store ---")
    # store.materialize_incremental(end_date=datetime.now())

    # print("\n--- Online features ---")
    # fetch_online_features(store)

    # print("\n--- Run feast teardown ---")
    # subprocess.run(["feast", "teardown"], cwd=repo_path)



def fetch_historical_features_entity_df(store: FeatureStore):
    # To get some valid unique_ids and timestamps, we would typically look at the data source.
    # For this test, we'll create a dummy entity dataframe.
    entity_df = pd.read_parquet("data/processed/test_data.parquet")
    entity_df["event_timestamp"] = pd.to_datetime(entity_df["event_timestamp"], utc = True)  
    entity_df["unique_id"] = entity_df["unique_id"].astype(str)
    train_df = store.get_historical_features(entity_df=entity_df, 
                                             features=["interface_traffic:y"]).to_df()
    print(train_df.head())


def fetch_online_features(store: FeatureStore):
    entity_rows = [
        {"unique_id": "1"},
    ]

    returned_features = store.get_online_features(
        features=[
            "interface_traffic:y",
            "lag_1h:lag_1h",
        ],
        entity_rows=entity_rows,
    ).to_dict()

    for key, value in sorted(returned_features.items()):
        print(key, " : ", value)


if __name__ == "__main__":
    run_demo()