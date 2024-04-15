# если что поменяй значения
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# 1. Создаем набор данных
x = np.arange(-20, 20, 0.1)
y = np.sin(x) + np.sin(np.sqrt(2)*x)

# 2. Разделяем данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=13)

# 3. Создаем и обучаем модель
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(1,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])

model.compile(loss='mse', optimizer='rmsprop', metrics=['mae'])
model.fit(X_train, y_train, epochs=1000, batch_size=32)

# 4. Оцениваем точность модели
_, accuracy = model.evaluate(X_train, y_train)
print("Точность на обучающей выборке:", accuracy)

_, accuracy2 = model.evaluate(X_test, y_test)
print("Точность на тестовой выборке:", accuracy2)

# 5. Визуализируем результаты
y_pred = model.predict(X_test)

plt.plot(x, y, label='f(x)')
plt.scatter(X_test, y_pred, label='Предсказания', color='red', alpha=1)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Аппроксимация функции f(x)')
plt.show()
