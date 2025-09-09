from neuralforecast import NeuralForecast
from src.utils.metrics import mape
from tqdm import tqdm
import pandas as pd
import mlflow
import os

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


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
        # Create a future dataframe for the next h steps
        
        # Predict h steps into the future from the end of the current history
        forecast = nf.predict(df=history_df)
        all_forecasts.append(forecast)

        # Update history by appending the *actual* observed data for the next iteration
        actuals_for_step = future_df.iloc[i: i + h]
        if actuals_for_step.empty:
            break

        # For models with exogenous variables, futr_df needs to contain their future values.
        # For models without, it just needs `unique_id` and `ds`.
        # `actuals_for_step` contains all necessary columns.
        forecast = nf.predict(futr_df=actuals_for_step)
        if forecast.empty:
            if not silent:
                print(f"Warning: Prediction for step starting at {actuals_for_step['ds'].min()} returned empty. Stopping.")
            break
        all_forecasts.append(forecast)

        history_df = pd.concat([history_df, actuals_for_step])
        print(f'Progress: {i}/{len(future_df)}')
        if not silent:
            print(f'Progress: {i+h}/{len(future_df)}')

    if not all_forecasts:
        return pd.DataFrame()
    return pd.concat(all_forecasts).reset_index()


def save_results(forecasts_df: pd.DataFrame, future_df: pd.DataFrame, output_dir: str, model_name: str, silent: bool) -> str:
    """
    Merges forecasts with actuals, saves to a CSV, and returns the file path.
    """
    if not silent:
        print("Processing and saving results...")

    # Merge with actuals from the future_df for comparison
    results_df = pd.merge(future_df, forecasts_df, on=['unique_id', 'ds'], how='left')
    results_df.dropna(inplace=True)  # Drop any rows where prediction wasn't possible

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    

    # Save CSV
    forecast_csv_path = os.path.join(output_dir, f'rolling_forecast_{model_name}.csv')
    results_df.to_csv(forecast_csv_path, index=False)
    return forecast_csv_path