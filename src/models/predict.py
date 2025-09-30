from neuralforecast import NeuralForecast
from tqdm import tqdm
import pandas as pd
def load_model(path="experiments/models/nhits_model.pth"):
    return NeuralForecast.load(path)

def forecast(model, data):
    return model.predict().reset_index()

def perform_rolling_forecast(nf: NeuralForecast, history_df: pd.DataFrame, future_df: pd.DataFrame, silent: bool) -> pd.DataFrame:
    """
    Performs a rolling forecast simulation.

    Args:
        nf: The initialized NeuralForecast object.
        history_df: The initial DataFrame to start forecasting from.
        future_df: The DataFrame with future actual values to iterate over.
        silent: Flag to disable progress bar.

    Returns:
        A DataFrame containing all concatenated forecasts.
    """
    h = nf.models[0].h  # Get forecast horizon from the loaded model
    all_forecasts = []

    if not silent:
        print(f"Performing rolling forecast with horizon h={h}...")

    # Use tqdm for a progress bar, which is disabled in silent mode
    for i in tqdm(range(0, len(future_df), h), desc="Rolling Forecast Steps", disable=silent):
        # Predict h steps into the future from the end of the current history
        forecast = nf.predict(df=history_df)
        all_forecasts.append(forecast)

        # Update history by appending the *actual* observed data for the next iteration
        actuals_for_step = future_df.iloc[i: i + h]
        if actuals_for_step.empty:
            break
        history_df = pd.concat([history_df, actuals_for_step])
        print(f'Progress: {i}/{len(future_df)}')
    return pd.concat(all_forecasts).reset_index()

if __name__ == "__main__":
    from src.utils.config import load_config
    import mlflow
    config = load_config("config.yaml")
    model = load_model(path="experiments/models/patchtst_model_0.17M_MSE")
    future_df = pd.read_parquet("data/processed/test_data.parquet")
    history_df = pd.read_parquet("data/processed/train_data.parquet")
    
    results = perform_rolling_forecast(
        future_df=future_df,
        history_df=history_df,
        nf=model,
        silent= False
    )
    print(results.head())