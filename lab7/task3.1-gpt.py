import numpy as np
import tensorflow as tf

# Подготовка данных
# Предположим, у нас есть временной ряд температуры
temperatures = [25.0, 26.0, 27.0, 28.0, 29.0]  # Пример данных
X_train = np.array([temperatures[i:i+3] for i in range(len(temperatures)-3)])
y_train = np.array([temperatures[i+3] for i in range(len(temperatures)-3)])

# Создание модели
model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(64, input_shape=(3, 1)),
    tf.keras.layers.Dense(1)
])

# Компиляция модели
model.compile(optimizer='adam', loss='mse')

# Обучение модели
model.fit(X_train, y_train, epochs=100, batch_size=1)

# Прогнозирование
next_temperature = model.predict(np.array([[27.0, 28.0, 29.0]]))
print("Next temperature prediction:", next_temperature)
