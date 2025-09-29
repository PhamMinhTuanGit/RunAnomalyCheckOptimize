import mlflow
from neuralforecast.models import NHITS, NBEATSx, PatchTST, TimesNet
from neuralforecast import NeuralForecast
from src.utils.logger import get_logger
from src.utils.metrics import mape
from neuralforecast.losses.pytorch import DistributionLoss, MAPE, MAE, MSE

logger = get_logger(__name__)


def count_parameters(model):
    """Counts the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model(model_name, config):
    """Factory function to create a model instance."""
    model_config = config["model"][model_name]
    h = model_config["h"]
    input_size = model_config["input_size"]
    max_steps = model_config["max_steps"]

    common_params = {
        "h": h,
        "input_size": input_size,
        "max_steps": max_steps,
        "batch_size": model_config.get("batch_size", 32),
        "learning_rate": float(model_config.get("lr", 1e-3)),
        "early_stop_patience_steps": model_config.get("early_stop_patience_steps", -1),
    }

    if model_name == "nhits":
        model = NHITS(
            **common_params,
            n_blocks=model_config.get("n_blocks", 3),
            mlp_units=model_config.get("mlp_units", [512, 512]),
            loss=DistributionLoss(distribution='Normal', level=[80, 90]),
        )
    elif model_name == "timesnet":
        model = TimesNet(
            **common_params,
            top_k=model_config['top_k'],
            num_kernels=model_config['num_kernels'],
            hidden_size=model_config["hidden_size"],
            conv_hidden_size=model_config["conv_hidden_size"],
            dropout=model_config["dropout"],
            loss=DistributionLoss(distribution='StudentT', level=[80, 85, 90, 95, 100]),
        )
    elif model_name == "patchtst":
        model = PatchTST(
            **common_params,
            n_heads=model_config['n_heads'],
            stride=model_config["stride"],
            hidden_size=model_config["hidden_size"],
            linear_hidden_size=model_config["linear_hidden_size"],
            patch_len=model_config["patch_len"],
            fc_dropout=model_config["fc_dropout"],
            attn_dropout=model_config["attn_dropout"],
            dropout=model_config["dropout"],
            encoder_layers=model_config["encoder_layers"],
            scaler_type=model_config["scaler_type"],
            revin=model_config["revin"],
            loss=MSE(),
        )
    else:
        raise ValueError(f"Model '{model_name}' not supported.")

    return model


def train_model(train_df, config, model_name):
    """Trains a model and logs the experiment with MLflow."""
    mlflow.set_experiment(f"{model_name.upper()} Experiment")

    with mlflow.start_run(run_name=config["experiment"]["run_name"], nested=True):
        model_config = config["model"][model_name]
        mlflow.log_params({
            "h": model_config["h"],
            "input_size": model_config["input_size"],
            "max_steps": model_config["max_steps"],
        })

        model = get_model(model_name, config)
        nf = NeuralForecast(models=[model], freq=config["data"]["freq"])

        logger.info(f"Training {model_name.upper()} model...")
        nf.fit(train_df[["unique_id", "ds", "y"]])

        n_params = count_parameters(nf.models[0])
        mlflow.log_metric("n_parameters", n_params)
        logger.info(f"Model has {n_params:,} trainable parameters")

        path = f"experiments/models/{model_name}_model_{n_params/1e6:.2f}M"
        if model_name == "patchtst":
            path += "_MSE"
        
        print(path)
        nf.save(path, overwrite=True)
        mlflow.log_artifact(path)

    return nf, n_params


if __name__ == "__main__":
    from src.utils.config import load_config
    import pandas as pd
    config = load_config("config.yaml")
    train_df = pd.read_parquet("data/processed/train_data.parquet").sort_values(['unique_id', 'ds'])
    
    # Example of training a PatchTST model
    model_to_train = "patchtst"
    models, n_params = train_model(train_df=train_df, config=config, model_name=model_to_train)