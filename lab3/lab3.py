import numpy as np
import pandas as pd
from src.neural_network import create_neural_network, train_neural_network,\
    DataGenerator, split_data, print_predictions_and_targets

# Генерация и сохранение данных
DataGenerator.generate_data('data.csv')

# Загрузка данных
data = pd.read_csv('data.csv')
train_data, test_data = split_data(data, train_fraction=0.8)

# Создание нейронной сети
num_layers = 2
num_neurons_per_layer = 3
num_inputs = 2

neural_net = create_neural_network(num_layers, num_neurons_per_layer, num_inputs)

# Обучение нейронной сети
epochs = 1000
learning_rate = 0.01
target_error = 0.001

train_neural_network(neural_net, train_data.values, epochs, learning_rate, target_error)

# Печать предсказаний на тестовых данных
print_predictions_and_targets(neural_net, test_data.values)
