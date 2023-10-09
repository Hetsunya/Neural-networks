import numpy as np
import pandas as pd
from src.neural_network import create_neural_network, train_neural_network,\
    DataGenerator, split_data, print_predictions_and_targets

# Генерация и сохранение данных
DataGenerator.generate_data('data.csv')

# Загрузка данных
data = pd.read_csv('data.csv')

# Перетасовка и разделение на обучающий и тестовый наборы
shuffled_data = data.sample(frac=1)
train_data, test_data = split_data(shuffled_data, train_fraction=0.8)

print(data)

print(train_data)

print(test_data)

neural_net = create_neural_network(2, 3, 2)

train_neural_network(neural_net, train_data.values, 100, 0.000001, 0.0001)

# Печать предсказаний на тестовых данных
print_predictions_and_targets(neural_net, test_data.values)

