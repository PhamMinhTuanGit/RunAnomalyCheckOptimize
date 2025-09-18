import numpy as np
from neuralforecast.losses.pytorch import MAPE, MAE, SMAPE, MSE
def mape(y_true, y_pred):
    return (np.abs((y_true - y_pred) / y_true)).mean() * 100

def get_metrics(y_true, y_pred, y_insample=None):
    metrics = {
        "MAE": MAE()(y_pred, y_true),
        "MSE": MSE()(y_pred, y_true, y_insample=y_insample) if y_insample is not None else MSE()(y_pred, y_true),
        "SMAPE": SMAPE()(y_pred, y_true, y_insample=y_insample) if y_insample is not None else SMAPE()(y_pred, y_true),
        "MAPE": MAPE()(y_pred, y_true)
    }
    return metrics

