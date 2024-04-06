import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def print_predictions_and_targets(neural_net, test_data):
    print("Предсказанные значения и целевые значения:")
    for inputs, target in test_data:
        predictions = [neuron.predict(inputs) for neuron in neural_net.neurons]
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

        # Обновляем веса и смещение
        self.weights += learning_rate * error * x
        self.bias += learning_rate * error

    @staticmethod
    def calculate_mse_for_network(neural_net, test_data):
        total_error = 0
        num_neurons = len(neural_net.neurons)

        for inputs, target in test_data:
            predictions = [neuron.predict(inputs) for neuron in neural_net.neurons]

            errors = [(1 / 2) * (target - prediction) ** 2 for prediction in predictions]
            total_error += sum(errors)

        # Усредняем ошибку по всем нейронам и элементам тестовой выборки
        mse = total_error / (num_neurons * len(test_data))
        return mse

    def get_weights(self):
        return self.weights, self.bias

class NeuralNetwork:
    def __init__(self, num_neurons, num_inputs_per_neuron):
        self.neurons = [Neuron(num_inputs_per_neuron) for _ in range(num_neurons)]

def create_neural_network(num_neurons, num_inputs_per_neuron):
    return NeuralNetwork(num_neurons, num_inputs_per_neuron)

def train_neural_network_1(neural_net, training_set, epochs, learning_rate, target_error):
    for epoch in range(epochs):
        total_error = 0
        for inputs, targets in training_set:
            for i, neuron in enumerate(neural_net.neurons):
                neuron.update_weights(inputs, targets[i], learning_rate)
                prediction = neuron.predict(inputs)
                total_error += (targets[i] - prediction) ** 2

        total_error /= len(training_set)

        if total_error < target_error:
            print(f"Обучение по первому варианту завершено на {epoch + 1}-й эпохе.")
            break

def train_neural_network_2(neural_net, training_set, epochs, learning_rate, target_error):
    for epoch in range(epochs):
        total_error = 0
        for inputs, target in training_set:
            for neuron in neural_net.neurons:
                prediction = neuron.predict(inputs)
                error = target - prediction
                for i in range(len(neuron.weights)):
                    neuron.weights[i] += learning_rate * error * inputs[i]
                total_error += error ** 2

        total_error /= len(training_set)

        if total_error < target_error:
            print(f"Обучение по второму варианту завершено на {epoch + 1}-й эпохе.")
            break

def split_data(data, train_fraction):
    num_samples = len(data)
    train_size = int(train_fraction * num_samples)
    train_data = data[:train_size]
    test_data = data[train_size:]
    return train_data, test_data