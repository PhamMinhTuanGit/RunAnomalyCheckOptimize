from src.utils.metrics import mape

def evaluate(y_true, y_pred):
    return {"MAPE": mape(y_true, y_pred)}
