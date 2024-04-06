import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def print_predictions_and_targets(neural_net, test_data):
    print("Предсказанные значения и целевые значения:")
    for row in test_data:
        inputs = row[:-1]
        target = row[-1]
        layer_inputs = inputs
        for layer in neural_net.layers:
            layer_outputs = layer.forward_pass(layer_inputs)
            layer_inputs = layer_outputs

        predictions = layer_outputs
        print(f"Предсказание: {predictions}, Целевое значение: {target}")

def plot_3d_graph(x1, x2, y, title, color='b'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x1, x2, y, c=color, marker='o')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Y')
    plt.title(title)
    plt.show()

class DataGenerator:
    @staticmethod
    def generate_data(filename):
        data = {'x1': list(range(1, 101)),
                'x2': list(range(100, 0, -1))}
        data['Y'] = [x1 * 3 + x2 * 8 for x1, x2 in zip(data['x1'], data['x2'])]
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)

    @staticmethod
    def add_noise(filename, noise_std):
        data = pd.read_csv(filename)
        noise = np.random.normal(0, noise_std, len(data))
        data['Y'] += noise
        data.to_csv(filename, index=False)

class Neuron:
    def backward_pass(self, inputs, target, learning_rate):
        inputs = np.array(inputs)
        prediction = self.predict(inputs)
        error = target - prediction

        # Поэлементное умножение
        self.weights += learning_rate * error * inputs
        self.bias += learning_rate * error

    def __init__(self, num_inputs):
        self.weights = np.random.uniform(0.001, 0.2, num_inputs)
        self.bias = np.random.uniform(0.001, 0.2)
        self.accuracy = 0.1


    def predict(self, x):
        summator = np.dot(x, self.weights) + self.bias
        return summator
    def update_weights(self, x, y, learning_rate):
        x = np.array(x)
        prediction = self.predict(x)
        error = y - prediction

        # Проверяем, не равны ли prediction и target
        if prediction != y:
            self.weights += learning_rate * error * x
            self.bias += learning_rate * error

    @staticmethod
    def calculate_mse_for_network(neural_net, test_data):
        total_error = 0
        num_neurons = sum(len(layer.neurons) for layer in neural_net.layers)

        for inputs, target in test_data:
            layer_inputs = inputs
            for layer in neural_net.layers:
                layer_outputs = layer.forward_pass(layer_inputs)
                layer_inputs = layer_outputs

            errors = [(1 / 2) * (target - output) ** 2 for output in layer_outputs]
            total_error += sum(errors)

        # Усредняем ошибку по всем нейронам и элементам тестовой выборки
        mse = total_error / (num_neurons * len(test_data))
        return mse

    def get_weights(self):
        return self.weights, self.bias

class Layer:
    def __init__(self, num_neurons, num_inputs):
        self.neurons = [Neuron(num_inputs) for _ in range(num_neurons)]

    def forward_pass(self, inputs):
        return [neuron.predict(inputs) for neuron in self.neurons]

    def backward_pass(self, inputs, target, learning_rate):
        for i, neuron in enumerate(self.neurons):
            neuron.update_weights(inputs, target, learning_rate)

class NeuralNetwork:
    def __init__(self, num_layers, num_neurons_per_layer, num_inputs):
        self.layers = [Layer(num_neurons_per_layer, num_inputs)]
        for _ in range(num_layers - 1):
            self.layers.append(Layer(num_neurons_per_layer, num_neurons_per_layer))

def create_neural_network(num_layers, num_neurons_per_layer, num_inputs):
    return NeuralNetwork(num_layers, num_neurons_per_layer, num_inputs)

def train_neural_network(neural_net, training_set, epochs, learning_rate, target_error):
    for epoch in range(epochs):
        total_error = 0
        for row in training_set:
            inputs = row[:-1]  # Первые (len(row) - 1) элементов - это входы
            target = row[-1]   # Последний элемент - это цель

            layer_inputs = inputs
            for layer in neural_net.layers:
                layer_outputs = layer.forward_pass(layer_inputs)
                layer_inputs = layer_outputs

            total_error += sum([(target - output) ** 2 for output in layer_outputs])

            for i in range(len(neural_net.layers) - 1, -1, -1):
                layer = neural_net.layers[i]
                layer_inputs = inputs if i == 0 else neural_net.layers[i - 1].forward_pass(inputs)
                for j, neuron in enumerate(layer.neurons):
                    neuron.backward_pass(layer_inputs, target, learning_rate)

        total_error /= len(training_set)

        if total_error < target_error:
            print(f"Обучение завершено на {epoch + 1}-й эпохе.")
            break

def split_data(data, train_fraction):
    num_samples = len(data)
    shuffled_data = data.sample(frac=1)  # Перетасовываем данные
    train_size = int(train_fraction * num_samples)
    train_data = shuffled_data[:train_size]
    test_data = shuffled_data[train_size:]
    return train_data, test_data

