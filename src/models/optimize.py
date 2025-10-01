
import optuna
import pandas as pd
import torch
import mlflow
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS, PatchTST, TimesNet
from src.utils.config import load_config
from src.models.train import get_model
from neuralforecast.losses.pytorch import MAPE
import logging

# Suppress informational messages from NeuralForecast
logging.getLogger("neuralforecast").setLevel(logging.WARNING)

def objective(trial, model_name: str, config: dict, train_df: pd.DataFrame):
    """
    Objective function for Optuna optimization.
    """
    # Define hyperparameter search space
    if model_name == "nhits":
        config['model']['nhits']['n_blocks'] = [trial.suggest_int("n_blocks", 1, 3)] * 3
        config['model']['nhits']['mlp_units'] = [[trial.suggest_int("mlp_units", 256, 1024)] * 2] * 3
        config['model']['nhits']['lr'] = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    elif model_name == "patchtst":
        config['model']['patchtst']['n_heads'] = trial.suggest_int("n_heads", 4, 16)
        config['model']['patchtst']['hidden_size'] = trial.suggest_categorical("hidden_size", [64, 128, 256, 512, 768])
        config['model']['patchtst']['patch_len'] = trial.suggest_int("patch_len", 8, 32)
        config['model']['patchtst']['encoder_layers'] = trial.suggest_int("encoder_layers", 2, 8)
        config['model']['patchtst']['lr'] = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    elif model_name == "timesnet":
        config['model']['timesnet']['top_k'] = trial.suggest_int("top_k", 1, 5)
        config['model']['timesnet']['num_kernels'] = trial.suggest_int("num_kernels", 1, 6)
        config['model']['timesnet']['hidden_size'] = trial.suggest_categorical("hidden_size", [64, 128, 256])
        config['model']['timesnet']['conv_hidden_size'] = trial.suggest_categorical("conv_hidden_size", [64, 128, 256])
        config['model']['timesnet']['lr'] = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    else:
        raise ValueError(f"Model '{model_name}' not supported for optimization.")

    # Get model
    model = get_model(model_name, config)

    # Setup NeuralForecast
    nf = NeuralForecast(models=[model], freq=config["data"]["freq"])

    # Cross-validation
    # Using a smaller number of validation windows for speed during optimization
    cv_df = nf.cross_validation(df=train_df, n_windows=5, step_size=config['model'][model_name]['h'])
    
    # Calculate MAPE
    mape_value = MAPE()(cv_df['y'], cv_df[model_name])
    
    return mape_value

if __name__ == "__main__":
    # Load config and data
    config = load_config("config.yaml")
    train_df = pd.read_parquet("data/processed/train_data.parquet").sort_values(['unique_id', 'ds'])

    models_to_optimize = ["nhits", "patchtst", "timesnet"]

    for model_name in models_to_optimize:
        print(f"Optimizing {model_name.upper()}...")
        
        # Create study
        study = optuna.create_study(direction="minimize")
        
        # Run optimization
        study.optimize(lambda trial: objective(trial, model_name, config, train_df), n_trials=20)
        
        # Print best trial
        best_trial = study.best_trial
        print(f"Best trial for {model_name.upper()}:")
        print(f"  Value (MAPE): {best_trial.value}")
        print("  Params: ")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")
