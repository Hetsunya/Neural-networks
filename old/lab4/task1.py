import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense

# 1. Создание набора данных
x = np.arange(-20, 20, 0.1)
y = x

# 2. Разделение на обучающую и тестовую выборки
split_idx = int(0.8 * len(x))
x_train, x_test = x[:split_idx], x[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# 3. Создание модели нейронной сети
model = Sequential()
model.add(Dense(units=1, input_dim=1))

# 4. Компиляция и обучение модели
model.compile(optimizer='sgd', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=10, batch_size=4)

# 5. Построение графика
plt.scatter(x, model.predict(x), color='r', label='Predictions')
plt.plot(x, y, label='True Function')
plt.legend()
plt.show()
