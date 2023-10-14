import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense

# Генерация данных для окружности
theta = np.linspace(0, 20*np.pi, 1000)
radius = 5.0
x_circle = radius * np.cos(theta)
y_circle = radius * np.sin(theta)

# Добавление шума к данным
noise = np.random.normal(0, 0.5, len(theta))
x_data = x_circle + noise
y_data = y_circle + noise

# Создание нейронной сети
model = Sequential()
model.add(Dense(units=10, input_dim=2, activation='relu'))
model.add(Dense(units=1, activation='linear'))

# Сборка модели
model.compile(optimizer='adam', loss='mean_squared_error')

# Стеки данных
data = np.column_stack((x_data, y_data))

# Разделение данных на обучающую и тестовую выборки
split_idx = int(0.8 * len(data))
train_data, test_data = data[:split_idx], data[split_idx:]

# Разделение x и y
x_train, y_train = train_data[:, 0:2], train_data[:, 1]
x_test, y_test = test_data[:, 0:2], test_data[:, 1]  # Было [:, 0:2], test_data[:, 2]

# Обучение модели
model.fit(x_train, y_train, epochs=100, batch_size=100)

# Оценка модели на тестовых данных
loss = model.evaluate(x_test, y_test)
print(f'Loss on test data: {loss}')

# Генерация предсказаний для построения графика
predictions = model.predict(x_test)

# Построение графика окружности и предсказаний
plt.scatter(x_test[:, 0], x_test[:, 1], c=predictions.flatten(), cmap='viridis')
plt.colorbar(label='Predictions')
plt.plot(x_circle, y_circle, color='r', label='True Circle')
plt.legend()
plt.show()
