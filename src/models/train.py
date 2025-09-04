import mlflow
from neuralforecast.models import NHITS, NBEATSx, PatchTST, TimesNet
from neuralforecast import NeuralForecast
from src.utils.logger import get_logger
from src.utils.metrics import mape
from neuralforecast.losses.pytorch import DistributionLoss 

logger = get_logger(__name__)


def count_parameters(model):
    """Đếm số lượng tham số trainable trong model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_nhits(train_df, test_df, config):
    mlflow.set_experiment("NHITS Experiment")

    with mlflow.start_run(run_name=config["experiment"]["run_name"]):
        # Hyperparameters từ config
        

        mlflow.log_params({
            "h": config["model"]["nhits"]["h"],
            "input_size": config["model"]["nhits"]["input_size"],
            "max_steps": config["model"]["nhits"]["max_steps"]

        })

        # Model
        models = [NHITS(
            h=config["model"]["nhits"]["h"],
            input_size=config["model"]["nhits"]["input_size"],
            max_steps=config["model"]["nhits"]["max_steps"],
            n_blocks=config["model"]["nhits"].get("n_blocks", 3),
            mlp_units=config["model"]["nhits"].get("mlp_units", [512, 512]),
            batch_size=config["model"]["nhits"].get("batch_size", 32),
            learning_rate=config["model"]["nhits"].get("learning_rate", 1e-3),
            early_stop_patience_steps=config["model"]["nhits"].get("early_stop_patience_steps", -1),
            loss=DistributionLoss(distribution='Normal', level=[80, 90]),
        )]

        mlflow.log_params(models[0].__dict__)

        nf = NeuralForecast(models=models, freq=config["data"]["freq"])

        logger.info("Training NHITS model...")
        nf.fit(train_df[["unique_id", "ds", "y"]])

        # Log số lượng tham số
        nhits_model = nf.models[0]
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

    return nf, score, n_params


def train_nbeatsx(train_df, test_df, config):
    # Hyperparameters từ config
    H = config["model"]["nbeatsx"]["h"]
    INPUT_SIZE = config["model"]["nbeatsx"]["input_size"]
    MAX_STEPS = config["model"]["nbeatsx"]["max_steps"]
    HIST_EXOG_LIST = config["model"]["nbeatsx"].get("hist_exog_list", ["lag_5min", "lag_30min", "lag_2h"])

    # Model
    models = [NBEATSx(
        h=H,
        input_size=INPUT_SIZE,
        max_steps=MAX_STEPS,
        n_blocks=config["model"]["nbeatsx"]['n_blocks'],
        mlp_units=config["model"]["nbeatsx"]["mlp_units"],
        batch_size=config["model"]["nbeatsx"]["batch_size"],
        learning_rate=float(config["model"]["nbeatsx"]["lr"]),
        early_stop_patience_steps=config["model"]["nbeatsx"]["early_stop_patience_steps"],
        hist_exog_list=HIST_EXOG_LIST,
        loss=DistributionLoss(distribution='Normal', level=[80, 90]),
        )]

    nf = NeuralForecast(models=models, freq=config["data"]["freq"])

    logger.info("Training NBEATSx model...")
    # Đảm bảo dataframe có đầy đủ các cột cần thiết: unique_id, ds, y, và hist_exog_list
    required_columns = ["unique_id", "ds", "y"] + HIST_EXOG_LIST
    nf.fit(train_df[required_columns])

    # Log số lượng tham số
    nbeatsx_model = nf.models[0]
    n_params = count_parameters(nbeatsx_model)
    logger.info(f"Model has {n_params:,} trainable parameters")

    # Forecast and evaluate
    logger.info("Forecasting with NBEATSx model...")
    # Đối với các mô hình có biến ngoại sinh, chúng ta cần cung cấp giá trị tương lai của các biến đó.
    # test_df chứa các giá trị tương lai này.
    forecast = nf.predict(futr_df=test_df).reset_index()

    y_true = test_df["y"].values[:len(forecast)]
    y_pred = forecast["NBEATSx"].values
    score = mape(y_true, y_pred)
    logger.info(f"Test MAPE for NBEATSx: {score:.2f}")

    nf.save("experiments/models/nbeatsx_model", overwrite=True)
    return nf, n_params, score

def train_timesnet(train_df, test_df, config):
    pass




def train_patchtst(train_df, test_df, config):
    # Hyperparameters từ config
    H = config["model"]["patchtst"]["h"]
    INPUT_SIZE = config["model"]["patchtst"]["input_size"]
    MAX_STEPS = config["model"]["patchtst"]["max_steps"]
    HIST_EXOG_LIST = config["model"]["patchtst"].get("hist_exog_list", ["lag_5min", "lag_30min", "lag_2h"])   
    # Model
    models = [PatchTST(
        h=H,
        input_size=INPUT_SIZE,
        max_steps=MAX_STEPS,
        n_heads=config["model"]["patchtst"]['n_heads'],
        batch_size=config["model"]["patchtst"]["batch_size"],
        stride =config["model"]["patchtst"]["stride"],
        patch_len =config["model"]["patchtst"]["patch_len"],
        learning_rate=float(config["model"]["patchtst"]["lr"]),
        scaler_type=config["model"]["patchtst"]["scaler_type"],
        revin = config["model"]["patchtst"]["revin"],
        early_stop_patience_steps=config["model"]["patchtst"]["early_stop_patience_steps"],
        loss=DistributionLoss(distribution='StudentT', level=[80, 90]),
        )]
    nf = NeuralForecast(models=models, freq=config["data"]["freq"])

    logger.info("Training PatchTST model...")
    # Đảm bảo dataframe có đầy đủ các cột cần thiết: unique_id, ds, y, và hist_exog_list
    required_columns = ["unique_id", "ds", "y"] + HIST_EXOG_LIST
    nf.fit(train_df[required_columns])

    # Log số lượng tham số
    nbeatsx_model = nf.models[0]
    n_params = count_parameters(nbeatsx_model)
    logger.info(f"Model has {n_params:,} trainable parameters")

    # Forecast and evaluate
    logger.info("Forecasting with NBEATSx model...")
    # Đối với các mô hình có biến ngoại sinh, chúng ta cần cung cấp giá trị tương lai của các biến đó.
    # test_df chứa các giá trị tương lai này.
    forecast = nf.predict(futr_df=test_df).reset_index()

    y_true = test_df["y"].values[:len(forecast)]
    y_pred = forecast["PatchTST-median"].values
    score = mape(y_true, y_pred)
    logger.info(f"Test MAPE for PatchTST: {score:.2f}")

    nf.save("experiments/models/patchtst_model", overwrite=True)
    return nf, n_params, score 