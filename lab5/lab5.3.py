import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Создание набора данных
x = np.arange(-20, 20, 0.1)
y = np.sin(x) + np.sin(np.sqrt(2)*x)

# Разделение данных и вычисление периода и смещения
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=13)
t_train = X_train // (2*np.pi)
fi_train = X_train % (2*np.pi)
t_test = X_test // (2*np.pi)
fi_test = X_test % (2*np.pi)

# Создание и обучение подсетей
# Сеть для предсказания периода (t)
model_period = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(1,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])

# Сеть для предсказания смещения (fi)
model_offset = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(1,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])

# Компиляция и обучение подсетей
model_period.compile(loss='mse', optimizer='adam')
model_offset.compile(loss='mse', optimizer='adam')

model_period.fit(X_train, t_train, epochs=150, batch_size=10, verbose=1)
model_offset.fit(X_train, fi_train, epochs=150, batch_size=10, verbose=1)

# Вариант 1: Объединение подсетей перед обучением
# Входной слой
input_layer = keras.Input(shape=(1,))

# Получение периода и смещения
period = model_period(input_layer)
offset = model_offset(input_layer)

# Объединение
combined = layers.Concatenate()([period, offset])

# Выходной слой
output_layer = layers.Dense(1)(combined)

# Создание и обучение сложной модели
model_complex1 = keras.Model(inputs=input_layer, outputs=output_layer)
model_complex1.compile(loss='mse', optimizer='rmsprop', metrics=['mae'])
model_complex1.fit(X_train, y_train, epochs=150, batch_size=10)

# Оценка точности
_, accuracy_complex1 = model_complex1.evaluate(X_test, y_test)
print("Точность сложной модели (объединение перед обучением):", accuracy_complex1)

# Вариант 2: Объединение подсетей после обучения
# Замораживаем веса подсетей
model_period.trainable = False
model_offset.trainable = False

# Входной слой
input_layer2 = keras.Input(shape=(1,))

# Получение периода и смещения
period2 = model_period(input_layer2)
offset2 = model_offset(input_layer2)

# Объединение
combined2 = layers.Concatenate()([period2, offset2])

# Выходной слой
output_layer2 = layers.Dense(1)(combined2)

# Создание и обучение сложной модели (только выходной слой)
model_complex2 = keras.Model(inputs=input_layer2, outputs=output_layer2)
model_complex2.compile(loss='mse', optimizer='rmsprop', metrics=['mae'])
model_complex2.fit(X_train, y_train, epochs=150, batch_size=10)

# Оценка точности
_, accuracy_complex2 = model_complex2.evaluate(X_test, y_test)
print("Точность сложной модели (объединение после обучения):", accuracy_complex2)

# Визуализация результатов
y_pred = model_complex2.predict(X_test)

plt.plot(x, y, label='f(x)')
plt.scatter(X_test, y_pred, label='Предсказания', color='red', alpha=1)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Аппроксимация функции f(x)')
plt.show()
