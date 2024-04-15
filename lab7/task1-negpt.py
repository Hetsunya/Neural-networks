import numpy as np

# Загрузка текста
text = open("war_and_peace.txt", "r", encoding="utf-8").read()
chars = sorted(list(set(text)))

# Создание словарей символ-индекс и индекс-символ
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}

# Подготовка данных для обучения
seq_length = 2  # Длина последовательности
dataX = []
dataY = []
for i in range(0, len(text) - seq_length, 1):
    seq_in = text[i:i + seq_length]
    seq_out = text[i + seq_length]
    dataX.append([char_to_ix[char] for char in seq_in])
    dataY.append(char_to_ix[seq_out])  # Убедитесь, что dataY содержит целые числа
n_patterns = len(dataX)

# One-hot кодирование
X = np.zeros((n_patterns, seq_length, len(chars)), dtype='float32')
y = np.zeros((n_patterns, len(chars)), dtype='float32')
for i, seq in enumerate(dataX):
    for t, char in enumerate(seq):
        X[i, t, char] = 1
    y[i, dataY[i]] = 1

# Параметры
hidden_size = 128
learning_rate = 0.01
epochs = 20
batch_size = 32

# Инициализация весов
Wx = np.random.randn(hidden_size, len(chars)) * 0.01
Wh = np.random.randn(hidden_size, hidden_size) * 0.01
Wy = np.random.randn(len(chars), hidden_size) * 0.01
bh = np.zeros((hidden_size, 1))
by = np.zeros((len(chars), 1))

# Функции RNN
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def forward_pass(x, h_prev, Wx, Wh, bh, by):
    x = x.T  # транспонируем x после one-hot кодирования
    h = np.tanh(np.dot(Wx, x) + np.dot(Wh, h_prev) + bh)
    y = sigmoid(np.dot(Wy, h) + by)
    return y, h


def backward_pass(x, y, h_prev, h, Wx, Wh, Wy, bh, by, learning_rate, y_true, dh_next):
    # Преобразуем y_true в one-hot вектор
    y_true_onehot = np.zeros_like(y)
    y_true_onehot[np.arange(y_true), y_true] = 1

    dy = y - y_true_onehot
    dWy = np.dot(dy, h.T)
    dby = dy
    dh_next = np.zeros_like(h)
    dh = np.dot(Wy.T, dy) + dh_next  # dh_next - градиент от следующего шага
    dh_raw = (1 - h * h) * dh
    dWx = np.dot(dh_raw, x.T)
    dWh = np.dot(dh_raw, h_prev.T)
    dbh = dh_raw

    # Обновление весов
    Wx -= learning_rate * dWx
    Wh -= learning_rate * dWh
    Wy -= learning_rate * dWy
    bh -= learning_rate * dbh
    by -= learning_rate * dby
    return dh_next

# Обучение
for epoch in range(epochs):
    for i in range(0, n_patterns, batch_size):
        batch_X = X[i:i + batch_size]
        batch_y = y[i:i + batch_size]
        for j in range(batch_size):
            x = batch_X[j]
            y_true = dataY[i+j]
            h_prev = np.zeros((hidden_size, 1))
            dh_next = np.zeros_like(h_prev)  # инициализация dh_next для каждого примера
            for t in range(seq_length):
                x_t = x[t].T  # транспонируем x до one-hot кодирования
                y, h = forward_pass(x_t, h_prev, Wx, Wh, bh, by)
                h_prev = h
                dh_next = backward_pass(x_t, y, h_prev, h, Wx, Wh, Wy, bh, by, learning_rate, y_true, dh_next)
    print(f"Epoch {epoch + 1}/{epochs}")

# Генерация текста
start_index = np.random.randint(0, len(text) - seq_length)
pattern = text[start_index: start_index + seq_length]
print("Seed:", pattern)

# Предсказание следующих символов
for i in range(10):
    x = np.zeros((seq_length, len(chars)))
    for t, char in enumerate(pattern):
        x[t, char_to_ix[char]] = 1

    y, _ = forward_pass(x[-1], h_prev, Wx, Wh, bh, by)
    index = np.argmax(y)
    result = ix_to_char[index]
    pattern += result

print("Generated text:", pattern)