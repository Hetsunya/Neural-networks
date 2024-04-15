import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense

# Создание обучающих данных
x_train = np.linspace(-20, 20, 1000)
y_train = np.abs(x_train)

# Создание и компиляция модели
model = Sequential()
model.add(Dense(10, input_dim=1, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mean_squared_error', optimizer='adam')

# Обучение модели
model.fit(x_train, y_train, epochs=100, batch_size=32)

# Визуализация результатов
x_test = np.linspace(-25, 25, 1000)
y_pred = model.predict(x_test)
plt.plot(x_train, y_train, label='Исходные данные')
plt.plot(x_test, y_pred, label='Предсказания модели')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Аппроксимация функции f(x) = |x|')
plt.legend()
plt.show()
