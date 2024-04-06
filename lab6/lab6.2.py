from PIL import Image
import os

# Путь к папке с данными
data_dir = "data6"

# Инициализация списка для хранения изображений и меток классов
images = []
labels = []

# Проходим по каждой папке в data6
for folder_name in os.listdir(data_dir):
    folder_path = os.path.join(data_dir, folder_name)

    # Проходим по каждому изображению в папке
    for file_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, file_name)

        # Открываем изображение с использованием контекстного управления
        with Image.open(image_path) as image:
            # Добавляем изображение в список
            images.append(image.copy())

            # Определяем метку класса (1 если это ваша функция, иначе 0) и добавляем в список меток
            if folder_name == "6":  # Замените "ваш_вариант" на соответствующую папку
                labels.append(1)
            else:
                labels.append(0)

# print("Загружено", len(images), "изображений.")
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam

# Предварительная обработка изображений и подготовка данных
def prepare_data(images, labels):
    # Преобразование изображений в массивы numpy
    images_array = np.array([img_to_array(image) for image in images])

    # Нормализация значений пикселей
    images_normalized = images_array / 255.0

    # Преобразование меток классов в бинарный формат
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    labels_categorical = to_categorical(labels_encoded)

    return images_normalized, labels_categorical

# Загрузка данных и подготовка обучающей и тестовой выборок
def load_data(images, labels, test_size=0.2):
    images_processed, labels_processed = prepare_data(images, labels)
    X_train, X_test, y_train, y_test = train_test_split(images_processed, labels_processed, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test

# Создание архитектуры нейронной сети
def create_model(input_shape):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    return model

# Загрузка данных
X_train, X_test, y_train, y_test = load_data(images, labels)

# Создание модели
input_shape = X_train[0].shape
model = create_model(input_shape)

# Компиляция модели
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])


# Обучение модели
model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))

# Оценка модели
loss, accuracy = model.evaluate(X_test, y_test)
print("Потери:", loss)
print("Точность:", accuracy)

import matplotlib.pyplot as plt

# Предсказание класса для тестовой выборки
predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(y_test, axis=1)

# Сохранение изображений с подписями
for i in range(len(X_test)):
    image = X_test[i]
    true_label = "Ваш вариант" if true_labels[i] == 1 else "Не ваш вариант"
    predicted_label = "Ваш вариант" if predicted_labels[i] == 1 else "Не ваш вариант"
    plt.imshow(image)
    plt.title(f"True: {true_label}, Predicted: {predicted_label}")
    plt.savefig(f"test_result_{i}.png")
    plt.close()

print("Результаты работы модели сохранены в файлы test_result_i.png")

