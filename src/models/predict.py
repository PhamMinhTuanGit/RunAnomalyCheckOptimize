from neuralforecast import NeuralForecast

def load_model(path="experiments/models/nhits_model.pth"):
    return NeuralForecast.load(path)

def forecast(model, data):
    return model.predict().reset_index()
