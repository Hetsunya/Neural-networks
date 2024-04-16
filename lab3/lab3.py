import numpy as np
import pandas as pd

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class Neuron:
    def __init__(self, num_inputs):
        self.weights = np.random.randn(num_inputs)
        self.bias = np.random.randn(1)

    def activate(self, inputs):
        self.z = np.dot(inputs.T, self.weights) + self.bias  # Исправленная строка
        self.a = sigmoid(self.z)
        return self.a


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_layer = [Neuron(input_size) for _ in range(hidden_size)]
        self.output_layer = [Neuron(hidden_size) for _ in range(output_size)]

    def feedforward(self, inputs):
        hidden_outputs = np.array([n.activate(inputs) for n in self.hidden_layer])
        outputs = np.array([n.activate(hidden_outputs) for n in self.output_layer])
        return outputs

    def backpropagate(self, inputs, targets, learning_rate):
        # Прямой проход
        hidden_outputs = np.array([n.activate(inputs) for n in self.hidden_layer])
        outputs = np.array([n.activate(hidden_outputs) for n in self.output_layer])

        # Ошибка выходного слоя
        output_errors = targets - outputs
        output_gradients = output_errors * sigmoid_derivative(outputs)

        # Ошибка скрытого слоя
        hidden_errors = np.dot(output_gradients, self.output_layer[0].weights.T)
        hidden_gradients = hidden_errors * sigmoid_derivative(hidden_outputs)

        # Обновление весов и смещений
        for i, n in enumerate(self.output_layer):
            n.weights += learning_rate * np.dot(hidden_outputs.T, output_gradients[i])
            n.bias += learning_rate * output_gradients[i]
        for i, n in enumerate(self.hidden_layer):
            n.weights += learning_rate * np.dot(inputs.T, hidden_gradients[i])
            n.bias += learning_rate * hidden_gradients[i]

# Загрузка данных
data = pd.read_csv("3lab_data.csv")
inputs = data[["x1", "x2", "x3"]].values
targets = data[["y1", "y2"]].values

# Создание сети
net = NeuralNetwork(3, 3, 2)

# Обучение
for epoch in range(10000):
    for i in range(len(inputs)):
        net.backpropagate(inputs[i], targets[i], learning_rate=0.1)

# Проверка весов
for n in net.hidden_layer + net.output_layer:
    print(n.weights)