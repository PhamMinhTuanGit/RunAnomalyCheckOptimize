import numpy as np

def mape(y_true, y_pred):
    return (np.abs((y_true - y_pred) / y_true)).mean() * 100
