import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense

# Создаем данные
X = np.arange(-20, 20, 0.1)
Y = X**2 + 2*X + 1

# Разделяем данные на обучающую и тестовую выборки
split_index = int(0.8 * len(X))
X_train, X_test = X[:split_index], X[split_index:]
Y_train, Y_test = Y[:split_index], Y[split_index:]

# Создаем модель нейронной сети
model = Sequential()
model.add(Dense(10, input_dim=1, activation='relu'))
model.add(Dense(1, activation='linear'))

# Компилируем модель
model.compile(loss='mean_squared_error', optimizer='adam')

# Обучаем модель
model.fit(X_train, Y_train, epochs=100, batch_size=10, verbose=2)

# Получаем предсказания на тестовой выборке
predictions = model.predict(X_test)

# Визуализация результатов
plt.scatter(X_test, predictions, color='red', label='Predictions')
plt.plot(X, X**2 + 2*X + 1, color='blue', label='True function')
plt.legend()
plt.show()
