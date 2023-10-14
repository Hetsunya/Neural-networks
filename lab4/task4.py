import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler

# 1. Создание набора данных
x = np.arange(-20 * np.pi, 20 * np.pi, 0.1)
y = np.sin(x)

# 2. Масштабирование данных
scaler = MinMaxScaler(feature_range=(-1, 1))
x_scaled = scaler.fit_transform(x.reshape(-1, 1))
y_scaled = scaler.fit_transform(y.reshape(-1, 1)).flatten()

# 3. Разделение на обучающую и тестовую выборки
split_idx = int(0.8 * len(x))
x_train, x_test = x_scaled[:split_idx], x_scaled[split_idx:]
y_train, y_test = y_scaled[:split_idx], y_scaled[split_idx:]

# 4. Создание более сложной модели
model = Sequential()
model.add(Dense(units=20, input_dim=1, activation='relu'))
model.add(Dense(units=20, activation='relu'))
model.add(Dense(units=1, activation='linear'))

# 5. Компиляция и обучение модели
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=10, batch_size=10, verbose=1)

# 6. Оценка модели на тестовых данных
loss = model.evaluate(x_test, y_test)
print(f'Loss on test data: {loss}')

# 7. Генерация предсказаний для построения графика
predictions = model.predict(x_test)

# 8. Построение графика
plt.scatter(x_test, y_test, color='b', label='True Function')
plt.scatter(x_test, predictions, color='r', label='Predictions')
plt.legend()
plt.show()
