import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense

# Генерация данных для окружности
theta = np.arange(-20, 20, 0.1)
radius = 5.0
x_data = radius * np.cos(theta)
y_data = radius * np.sin(theta)

# Стеки данных
data = np.column_stack((x_data, y_data))

# Разделение данных на обучающую и тестовую выборки
split_idx = int(0.8 * len(data))
train_data, test_data = data[:split_idx], data[split_idx:]

# Разделение x и y
x_train, y_train, x_test, y_test = train_data[:, 0:2], train_data[:, 1], test_data[:, 0:2], test_data[:, 1]

model = Sequential()
model.add(Dense(units=5, input_dim=2, activation='relu'))
model.add(Dense(units=1, activation='linear'))

# Сборка модели
model.compile(optimizer='adam', loss='mean_squared_error')

# Обучение модели
model.fit(x_train, y_train, epochs=10, batch_size=32)

# Генерация предсказаний для построения графика
predictions = model.predict(x_test)

# Построение графика окружности и предсказаний
plt.scatter(x_test[:, 0], x_test[:, 1], c=np.ones_like(predictions.flatten()), cmap='viridis')
plt.plot(0, 0, color='b', label='Predictions')
plt.plot(x_data, y_data, color='r', label='True Circle')
plt.legend()
plt.show()
