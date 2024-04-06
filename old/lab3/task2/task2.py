import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Загрузка данных
train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")
gender_submission = pd.read_csv("data/gender_submission.csv")

# Обработка категориальных признаков в исходных данных
train_data = pd.get_dummies(train_data, columns=['Embarked'], prefix='Embarked')
test_data = pd.get_dummies(test_data, columns=['Embarked'], prefix='Embarked')


# Предобработка данных
def preprocess_data(data):
    # Заполнение пропущенных значений
    data['Age'].fillna(data['Age'].median(), inplace=True)
    data['Fare'].fillna(data['Fare'].median(), inplace=True)

    # Преобразование категориальных признаков в числовые
    data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})

    # Отбор признаков
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
    if 'Embarked' in data.columns:
        features += ['Embarked']

    X = data[features]

    return X


# Подготовка данных
X_train = preprocess_data(train_data)
y_train = train_data['Survived']
X_test = preprocess_data(test_data)

# Нормализация данных
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Разделение данных на тренировочные и валидационные
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Создание нейронной сети
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Компиляция модели
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Обучение модели
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val))

# Вывод точности модели
accuracy = model.evaluate(X_val, y_val)[1]
print(f"\nТочность модели на валидационных данных: {accuracy * 100:.2f}%")

# Предсказания на тестовых данных
predictions = model.predict(X_test)
predictions = (predictions > 0.5).astype(int)

# Создание DataFrame для сохранения результатов
results = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': predictions.flatten()})

# Сохранение результатов в CSV файл
results.to_csv('titanic_predictions.csv', index=False)

# Сравнение с gender_submission
comparison = pd.merge(results, gender_submission, on='PassengerId', how='inner', suffixes=('_predicted', '_actual'))

# Вывод строк, где значения отличаются
differences = comparison[comparison['Survived_predicted'] != comparison['Survived_actual']]
print("\nРазличия между предсказаниями и фактическими значениями:")
print(differences)

import matplotlib.pyplot as plt

# Получение истории точности из обучения
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

# Построение графика точности
plt.plot(train_accuracy, label='Точность на тренировочных данных')
plt.plot(val_accuracy, label='Точность на валидационных данных')
plt.xlabel('Эпохи')
plt.ylabel('Точность')
plt.legend()
plt.show()