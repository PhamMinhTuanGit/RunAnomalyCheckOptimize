from neuralforecast.models import NBEATSx
from neuralforecast import NeuralForecast
from src.utils.logger import get_logger
from src.utils.metrics import mape
from neuralforecast.losses.pytorch import DistributionLoss
from src.utils.config import load_config
from src.models.train import count_parameters
from feast import FeatureStore
import pandas as pd

config = load_config("config.yaml")
train_config = load_config('config/nbeatsx.yaml')

df = pd.read_parquet("data/processed/train_data.parquet")
print(df.head())



models = [NBEATSx(
            h=train_config['h'],
            input_size=72,
            max_steps=100,
            n_blocks=train_config['n_blocks'],
            mlp_units=train_config["mlp_units"],
            batch_size=train_config["batch_size"],
            learning_rate=float(train_config["lr"]),
            early_stop_patience_steps=train_config["early_stop_patience_steps"],
            hist_exog_list=train_config['hist_exog_list'],
            loss=DistributionLoss(distribution='Normal', level=[80, 90])
            )]
nf = NeuralForecast(models=models, freq='5min')
nbeatsx_model = nf.models[0]
print(count_parameters(nbeatsx_model))

logger = get_logger("Train NBEATSx")
nf.fit(df=df)