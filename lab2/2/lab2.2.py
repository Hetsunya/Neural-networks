import pandas as pd
from src.neural_network import create_neural_network, train_neural_network_1, split_data, Neuron, print_predictions_and_targets

data = pd.read_csv('2lab_data.csv')
input_features = data[['x1', 'x2', 'x3', 'x4', 'x5', 'x6']]
targets = data[['y1', 'y2', 'y3']]  # Здесь берем все три целевых значения y1, y2, y3

training_set = list(zip(input_features.values, targets.values))
train_data, test_data = split_data(training_set, train_fraction=0.8)

neural_net = create_neural_network(3, 6)

train_neural_network_1(neural_net, train_data, epochs=1000, learning_rate=0.0001, target_error=0.01)

# Оценим ошибку на тестовой выборке для сети с 3 нейронами
# mse = Neuron.calculate_mse_for_network(neural_net, test_data)

# print(f"Ошибка на тестовой выборке для сети с 3 нейронами: {mse}")
#
# print("\nВеса и смещения после обучения для сети с 3 нейронами:")
# for i, neuron in enumerate(neural_net.neurons):
#     neuron_weights, neuron_bias = neuron.get_weights()
#     print(f"Нейрон {i + 1} - Веса: {neuron_weights}, Смещение: {neuron_bias}")

print(training_set)
print(input_features)
print(targets)

print_predictions_and_targets(neural_net, training_set)
# for input, target in training_set:
#     for neuron in range(len(neural_net.neurons)):
#         print(f"{neural_net.neurons[neuron].predict(input)} = {target}")

