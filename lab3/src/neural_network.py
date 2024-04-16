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
    def __init__(self, input_size):
        self.weights = np.random.rand(input_size)
        self.bias = np.random.rand()

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = self.sigmoid(np.dot(self.weights, inputs) + self.bias)
        return self.output

    def backward(self, error):
        # Calculate gradient
        self.error = error
        self.gradient = self.error * self.output * (1 - self.output)

        # Update weights and bias
        self.weights_gradient = self.inputs * self.gradient
        self.bias_gradient = self.gradient

        # Pass gradient to previous layer
        self.error_propagation = self.weights * self.gradient

        return self.weights_gradient, self.bias_gradient, self.error_propagation

class Layer:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.neurons = [Neuron(input_size) for _ in range(output_size)]

    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = np.array([neuron.forward(inputs) for neuron in self.neurons])
        return self.outputs

    def backward(self, errors):
        # Calculate gradients for each neuron
        weights_gradients = []
        bias_gradients = []
        errors_propagation = []
        for neuron, error in zip(self.neurons, errors):
            weights_gradient, bias_gradient, error_propagation = neuron.backward(error)
            weights_gradients.append(weights_gradient)
            bias_gradients.append(bias_gradient)
            errors_propagation.append(error_propagation)

        # Sum gradients for all neurons in the layer
        weights_gradients = np.sum(weights_gradients, axis=0)
        bias_gradients = np.sum(bias_gradients, axis=0)
        errors_propagation = np.sum(errors_propagation, axis=0)

        return weights_gradients, bias_gradients, errors_propagation

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_layer = Layer(input_size, hidden_size)
        self.output_layer = Layer(hidden_size, output_size)

    def forward(self, inputs):
        hidden_output = self.hidden_layer.forward(inputs)
        output = self.output_layer.forward(hidden_output)
        return output

    def backward(self, output_errors):
        hidden_errors = self.output_layer.backward(output_errors)
        input_errors = self.hidden_layer.backward(hidden_errors)
        return input_errors

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

