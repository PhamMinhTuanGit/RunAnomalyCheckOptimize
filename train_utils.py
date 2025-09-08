import sys
import os

# Chặn toàn bộ stdout tạm thời
sys.stdout = open(os.devnull, 'w')

# Import hoặc chạy NeuralForecast
from neuralforecast import NeuralForecast
# ... hoặc đoạn mã gây ra log

# Khôi phục stdout
sys.stdout = sys.__stdout__
