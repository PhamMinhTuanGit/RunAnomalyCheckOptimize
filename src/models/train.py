import mlflow
from neuralforecast.models import NHITS
from neuralforecast import NeuralForecast
from src.utils.logger import get_logger
from src.utils.metrics import mape

logger = get_logger(__name__)


def count_parameters(model):
    """Đếm số lượng tham số trainable trong model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_nhits(train_df, test_df, config):
    mlflow.set_experiment(config["experiment"]["name"])

    with mlflow.start_run(run_name=config["experiment"]["run_name"]):
        # Hyperparameters từ config
        H = config["model"]["h"]
        INPUT_SIZE = config["model"]["input_size"]
        MAX_STEPS = config["model"]["max_steps"]

        mlflow.log_params({
            "h": H,
            "input_size": INPUT_SIZE,
            "max_steps": MAX_STEPS

        })

        # Model
        models = [NHITS(
            h=config["model"]["h"],
            input_size=config["model"]["input_size"],
            max_steps=config["model"]["max_steps"],
            n_blocks=config["model"].get("n_blocks", 3),
            mlp_units=config["model"].get("mlp_units", [512, 512]),
            batch_size=config["model"].get("batch_size", 32),
            learning_rate=config["model"].get("learning_rate", 1e-3),
            early_stop_patience_steps=config["model"].get("early_stop_patience_steps", -1),
        )]

        mlflow.log_params(models[0].__dict__)

        nf = NeuralForecast(models=models, freq=config["data"]["freq"])

        logger.info("Training NHITS model...")
        nf.fit(train_df[["unique_id", "ds", "y"]])

        # Log số lượng tham số
        nhits_model = nf.models[0].model  
        n_params = count_parameters(nhits_model)
        mlflow.log_metric("n_parameters", n_params)
        logger.info(f"Model has {n_params:,} trainable parameters")

        # Forecast
        forecast = nf.predict().reset_index()
        y_true = test_df["y"].values[:len(forecast)]
        y_pred = forecast["NHITS"].values
        score = mape(y_true, y_pred)

        logger.info(f"Test MAPE: {score:.2f}")
        mlflow.log_metric("MAPE", score)

        # Save model
        nf.save("experiments/models/nhits_model")
        mlflow.log_artifact("experiments/models/nhits_model")

    return nf, score
