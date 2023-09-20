class Perceptron:
    def __init__(self):
        self.weights = [0.3, 0.2]
        self.trainingSet = [[1, 1, 1], [1, 0, 1], [0, 1, 1], [0, 0, 0]]
        self.accuracy = 0.1

    def train(self):
        for i in range(len(self.trainingSet)):
            d = self.trainingSet[i][2] - self.predict([self.trainingSet[i][0], self.trainingSet[i][1]])
            if d != 0:
                self.weights[0] += d * self.trainingSet[i][0] * self.accuracy
                self.weights[1] += d * self.trainingSet[i][1] * self.accuracy

    def test(self):
        if (self.predict([0,1]) == 1 and
            self.predict([1,0]) == 1 and
            self.predict([1,1]) == 1 and
            self.predict([0,0]) == 0):
            return 1
        return 0

    def predict(self, x):
        summator = x[0] * self.weights[0] + x[1] * self.weights[1]
        if summator >= 1:
            return 1
        else:
            return 0

    def get_weight(self):
        return self.weights

neuron = Perceptron()
print(f"Начальные веса = {neuron.get_weight()}")


while(neuron.test() != 1):
    neuron.train()
    print(f"Веса после обучения {neuron.get_weight()}")

