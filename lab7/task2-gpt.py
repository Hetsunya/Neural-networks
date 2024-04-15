from keras.models import Sequential
from keras.layers import SimpleRNN, Dense
from keras.optimizers import Adam
import numpy as np

# Подготовка данных
# Предположим, у нас есть временной ряд и мы хотим предсказать следующее значение
# Создаем временной ряд
time_series = np.sin(np.arange(200) * 0.1) + np.random.randn(200) * 0.1
# Формируем признаки и метки
X = np.reshape(time_series[:-1], (len(time_series)-1, 1, 1))
y = time_series[1:]

# Создание модели
model = Sequential()
model.add(SimpleRNN(units=64, input_shape=(1, 1)))
model.add(Dense(units=1))

# Компиляция модели
model.compile(optimizer=Adam(), loss='mse')

# Обучение модели
model.fit(X, y, epochs=20, batch_size=1)

# Прогнозирование
predictions = model.predict(X)

# Вывод результатов
print(predictions)
