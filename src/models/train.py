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

def train_nhits(train_df, config):
    mlflow.set_experiment("NHITS Experiment")

    with mlflow.start_run(run_name=config["experiment"]["run_name"], nested=True):
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

        nf = NeuralForecast(models=models, freq=config["data"]["freq"])

        logger.info("Training NHITS model...")
        nf.fit(train_df[["unique_id", "ds", "y"]])

        # Log số lượng tham số
        nhits_model = nf.models[0]
        n_params = count_parameters(nhits_model)
        mlflow.log_metric("n_parameters", n_params)
        logger.info(f"Model has {n_params:,} trainable parameters")

        # Save model
        nf.save("experiments/models/nhits_model", overwrite=True)

        # Create and log input example for the model signature
        

        mlflow.log_artifact("experiments/models/nhits_model")
    return nf, n_params

def train_nbeatsx(train_df, config):
    # Hyperparameters từ config
    mlflow.set_experiment("N-BEATsx Experiment")
    with mlflow.start_run(run_name=config["experiment"]["run_name"]):
        H = config["model"]["nbeatsx"]["h"]
        INPUT_SIZE = config["model"]["nbeatsx"]["input_size"]
        MAX_STEPS = config["model"]["nbeatsx"]["max_steps"]
        HIST_EXOG_LIST = config["model"]["nbeatsx"].get("hist_exog_list", ["lag_5min", "lag_30min", "lag_2h"])
        mlflow.log_params({
            "h": H,
            "input_size": INPUT_SIZE,
            "max_steps": MAX_STEPS

        })
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
        mlflow.log_metric("n_parameters", n_params)

        # Create and log input example for the model signature
        if nf.dataset:
            # nf.dataset[0] returns a tuple like (input_dict, target_tensor, ...).
            # We only need the first element for the model signature.
            input_example = nf.dataset[0][0]
            mlflow.pytorch.log_model(models[0], "model", input_example=input_example)

        nf.save("experiments/models/nbeatsx_model", overwrite=True)
        mlflow.log_artifact("experiments/models/nbeatsx_model")
    return nf, n_params

def train_timesnet(train_df, config):
    # Hyperparameters từ config
    H = config["model"]["timesnet"]["h"]
    INPUT_SIZE = config["model"]["timesnet"]["input_size"]
    MAX_STEPS = config["model"]["timesnet"]["max_steps"]

    mlflow.set_experiment("TimesNet Experiment")
    with mlflow.start_run(run_name=config["experiment"]["run_name"], nested=True):
        mlflow.log_params({
            "h": H,
            "input_size": INPUT_SIZE,
            "max_steps": MAX_STEPS
        })

        models = [TimesNet(
            h=H,
            input_size=INPUT_SIZE,
            max_steps=MAX_STEPS,
            top_k=config["model"]["timesnet"]['top_k'],
            num_kernels=config["model"]["timesnet"]["num_kernels"],
            d_model=config["model"]["timesnet"]["d_model"],
            d_ff=config["model"]["timesnet"]["d_ff"],
            e_layers=config["model"]["timesnet"]["e_layers"],
            dropout=config["model"]["timesnet"]["dropout"],
            learning_rate=float(config["model"]["timesnet"]["lr"]),
            batch_size=config["model"]["timesnet"]["batch_size"],
            early_stop_patience_steps=config["model"]["timesnet"]["early_stop_patience_steps"],
            loss=DistributionLoss(distribution='StudentT', level=[80, 90]),
        )]
        nf = NeuralForecast(models=models, freq=config["data"]["freq"])

        logger.info("Training TimesNet model...")
        required_columns = ["unique_id", "ds", "y"]
        nf.fit(train_df[required_columns])

        # Log số lượng tham số
        timesnet_model = nf.models[0]
        n_params = count_parameters(timesnet_model)
        logger.info(f"Model has {n_params:,} trainable parameters")
        mlflow.log_metric("n_parameters", n_params)

        # Create and log input example for the model signature
        if nf.dataset:
            # nf.dataset[0] returns a tuple like (input_dict, target_tensor, ...).
            # We only need the first element for the model signature.
            input_example = nf.dataset[0][0]
            mlflow.pytorch.log_model(models[0], "model", input_example=input_example)

        # Lưu model
        nf.save("experiments/models/timesnet_model", overwrite=True)
        mlflow.log_artifact("experiments/models/timesnet_model")

    return nf, n_params


def train_patchtst(train_df, config):
    # Hyperparameters từ config
    H = config["model"]["patchtst"]["h"]
    INPUT_SIZE = config["model"]["patchtst"]["input_size"]
    MAX_STEPS = config["model"]["patchtst"]["max_steps"] 
    # Model
    mlflow.set_experiment("PatchTST Experiment")
    with mlflow.start_run(run_name=config["experiment"]["run_name"], nested=True):
        mlflow.log_params({
            "h": H,
            "input_size": INPUT_SIZE,
            "max_steps": MAX_STEPS

        })
        models = [PatchTST(
            h=H,
            input_size=INPUT_SIZE,
            max_steps=MAX_STEPS,
            n_heads=config["model"]["patchtst"]['n_heads'],
            batch_size=config["model"]["patchtst"]["batch_size"],
            stride =config["model"]["patchtst"]["stride"],
            hidden_size=config["model"]["patchtst"]["hidden_size"],
            linear_hidden_size=config["model"]["patchtst"]["linear_hidden_size"],
            patch_len =config["model"]["patchtst"]["patch_len"],
            fc_dropout=config["model"]["patchtst"]["fc_dropout"],
            attn_dropout=config["model"]["patchtst"]["attn_dropout"],
            dropout=config["model"]["patchtst"]["dropout"],
            encoder_layers=config["model"]["patchtst"]["encoder_layers"],
            learning_rate=float(config["model"]["patchtst"]["lr"]),
            scaler_type=config["model"]["patchtst"]["scaler_type"],
            revin = config["model"]["patchtst"]["revin"],
            early_stop_patience_steps=config["model"]["patchtst"]["early_stop_patience_steps"],
            loss=DistributionLoss(distribution='StudentT', level=[90, 100]),
            )]
        nf = NeuralForecast(models=models, freq=config["data"]["freq"])

        logger.info("Training PatchTST model...")
        # Đảm bảo dataframe có đầy đủ các cột cần thiết: unique_id, ds, y, và hist_exog_list
        
        required_columns = ["unique_id", "ds", "y"]
        nf.fit(train_df[required_columns])

        # Log số lượng tham số
        patchtst_model = nf.models[0]
        n_params = count_parameters(patchtst_model)
        logger.info(f"Model has {n_params:,} trainable parameters")
        mlflow.log_metric("n_parameters", n_params)
        # Forecast and evaluate
        
        
        nf.save("experiments/models/patchtst_model", overwrite=True)
        mlflow.log_artifact("experiments/models/patchtst_model")
    return nf, n_params