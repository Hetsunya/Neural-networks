import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

# Создание набора данных
X = np.arange(-20, 20, 0.1)
y = np.sin(X)

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание модели
model = Sequential()
model.add(Dense(200, input_dim=1, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='linear'))

# Компиляция модели
model.compile(loss='mean_squared_error', optimizer='adam')

# Список для сохранения значений функции потерь
history = []

# Обучение модели с сохранением истории
history = model.fit(X_train, y_train, epochs=1000, batch_size=32, verbose=1)

# Получение предсказаний для всего диапазона значений X
y_pred = model.predict(X)

# Визуализация результатов
plt.figure(figsize=(12, 6))

# График функции потерь
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'])
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')

# График предсказаний и истинной функции
plt.subplot(1, 2, 2)
plt.scatter(X, y, color='black', label='Actual data')
plt.scatter(X, y_pred, color='red', label='Predicted data')
plt.plot(X, np.sin(X), color='blue', linewidth=3, label='True function')
plt.xlabel('X')
plt.ylabel('f(X)')
plt.legend()

plt.tight_layout()
plt.show()
