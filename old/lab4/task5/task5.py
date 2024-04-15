# Установка библиотек
# pip install scikit-surprise pandas

import pandas as pd
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split
from surprise import accuracy

# Загрузка данных из CSV файла
file_path = 'spotify_songs.csv'
data = pd.read_csv(file_path)

# Определение рейтингов
reader = Reader(rating_scale=(0, 100))  # Предположим, что 'track_popularity' находится в диапазоне от 0 до 100

# Создание объекта Dataset
dataset = Dataset.load_from_df(data[['playlist_id', 'track_id', 'track_popularity']], reader)

# Разбивка на обучающий и тестовый наборы
trainset, testset = train_test_split(dataset, test_size=0.2)

# Обучение модели
model = KNNBasic(sim_options={'user_based': True})
model.fit(trainset)

# Получение предсказаний
predictions = model.test(testset)

# Оценка качества модели
rmse = accuracy.rmse(predictions)
print(f'RMSE: {rmse}')
