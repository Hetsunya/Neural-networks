import numpy as np


class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Инициализация параметров сети
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Why = np.random.randn(output_size, hidden_size) * 0.01
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))
        self.h_prev = np.zeros((hidden_size, 1))

    def forward(self, x):
        self.h_next = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, self.h_prev) + self.bh)
        y = np.dot(self.Why, self.h_next) + self.by
        return y, self.h_next

    def backward(self, x, y_true, learning_rate=0.01):
        # Прямое распространение
        y_pred, _ = self.forward(x)

        # Обратное распространение
        dy = y_pred - y_true
        dWhy = np.dot(dy, self.h_next.T)
        dby = dy
        dh_next = np.dot(self.Why.T, dy)
        dh_raw = (1 - self.h_next * self.h_next) * dh_next
        dbh = dh_raw
        dWxh = np.dot(dh_raw, x.T)
        dWhh = np.dot(dh_raw, self.h_prev.T)

        # Обновление параметров
        self.Wxh -= learning_rate * dWxh
        self.Whh -= learning_rate * dWhh
        self.Why -= learning_rate * dWhy
        self.bh -= learning_rate * dbh
        self.by -= learning_rate * dby

        # Запоминаем предыдущее скрытое состояние
        self.h_prev = self.h_next

    def train(self, X_train, y_train, epochs=100, learning_rate=0.01):
        for epoch in range(epochs):
            for x, y_true in zip(X_train, y_train):
                self.backward(x, y_true, learning_rate)

    def predict(self, X_test):
        predictions = []
        for x in X_test:
            y_pred, _ = self.forward(x)
            predictions.append(y_pred)
        return predictions


# Пример использования
input_size = 3
hidden_size = 4
output_size = 2

# Создание экземпляра сети
rnn = SimpleRNN(input_size, hidden_size, output_size)

# Генерация некоторых входных и выходных данных для обучения
X_train = [np.random.randn(input_size, 1) for _ in range(100)]
y_train = [np.random.randn(output_size, 1) for _ in range(100)]

# Обучение сети
rnn.train(X_train, y_train)

# Пример использования обученной сети для предсказания
X_test = [np.random.randn(input_size, 1) for _ in range(10)]
predictions = rnn.predict(X_test)
print(predictions)
