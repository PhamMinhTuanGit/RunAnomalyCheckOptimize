import numpy as np
from neuralforecast.losses.pytorch import MAPE, MAE, SMAPE, MSE
import pandas as pd
def mape(y_true, y_pred):
    return (np.abs((y_true - y_pred) / y_true)).mean() * 100
y_insample=pd.read_parquet("data/processed/train_data.parquet").sort_values(['unique_id', 'ds'])
def get_metrics(y_true, y_pred, y_insample=y_insample):
    metrics = {
        "MAE": MAE()(y_pred, y_true),
        "MSE": MSE()(y_pred, y_true, y_insample=y_insample) if y_insample is not None else MSE()(y_pred, y_true),
        "SMAPE": SMAPE()(y_pred, y_true, y_insample=y_insample) if y_insample is not None else SMAPE()(y_pred, y_true),
        "MAPE": MAPE()(y_pred, y_true, y_insample=y_insample) if y_insample is not None else MAPE()(y_pred, y_true),
    }
    return metrics

