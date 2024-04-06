import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Создаем набор данных
x = np.arange(-20, 20, 0.1)
y = np.sin(x) + np.sin(np.sqrt(2)*x)

# Разделяем данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=13)

# Добавляем переменные номера периода и смещения внутри периодически повторяющегося участка кривой
t_train = X_train // (2*np.pi)
fi_train = X_train % (2*np.pi)

t_test = X_test // (2*np.pi)
fi_test = X_test % (2*np.pi)

# Создаем и обучаем модель
model = keras.Sequential([
    layers.Dense(32, activation='relu', input_shape=(2,)), # Два входа: номер периода и смещение
    layers.Dense(16, activation='relu'),
    layers.Dense(1)
])

model.compile(loss='mse', optimizer='rmsprop', metrics=['mae'])
model.fit(np.column_stack((t_train, fi_train)), y_train, epochs=300, batch_size=50)

# Оцениваем точность модели
_, accuracy = model.evaluate(np.column_stack((t_train, fi_train)), y_train)
print("Точность на обучающей выборке:", accuracy)

_, accuracy2 = model.evaluate(np.column_stack((t_test, fi_test)), y_test)
print("Точность на тестовой выборке:", accuracy2)

# Визуализируем результаты
y_pred = model.predict(np.column_stack((t_test, fi_test)))

plt.plot(x, y, label='f(x)')
plt.scatter(X_test, y_pred, label='Предсказания', color='red', alpha=1)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Аппроксимация функции f(x)')
plt.show()
