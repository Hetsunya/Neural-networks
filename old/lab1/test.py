import numpy as np
# Для дизъюкции веса дожны быть > 1
class Perceptron:
    def __init__(self, weights):
        self.weights = 0.1
        self.last_error = 1.1
        self.smoothing = 0.0001

    def get_weight(self):
        return self.weights
    def get_last_error(self):
        return self.last_error
    def get_smoothing(self):
        return self.smoothing

    def procces_data(self, inpud_data):
        return inpud_data * self.weights

    def train(self, input, execpetResult):
        result_now = input * self.weights

        self.last_error = execpetResult - result_now
        correction = self.last_error / result_now
        correction *= self.smoothing

        self.weights += correction

    def check_training(self):
        if (self.last_error > self.smoothing or self.last_error < -self.last_error):
            return True
        else:
            return False

neuron = Perceptron()

inputData = 228
excpectedResult = 1337
print(neuron.procces_data(inputData))

iteration = 1
while(neuron.check_training()):
    neuron.train(inputData, excpectedResult)
    iteration = iteration + 1

print(neuron.get_weight())
print(neuron.procces_data(inputData))