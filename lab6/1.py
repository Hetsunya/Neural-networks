from PIL import Image
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical


# Путь к папке с данными
data_dir = "data6"

# Инициализация списка для хранения изображений и меток классов
images = []
labels = []

# Размер пакета для обработки
batch_size = 500

# Проходим по каждой папке в data6
for folder_name in os.listdir(data_dir):
    folder_path = os.path.join(data_dir, folder_name)

    # Проходим по каждому изображению в папке
    for i, file_name in enumerate(os.listdir(folder_path)):
        image_path = os.path.join(folder_path, file_name)

        # Открываем изображение с использованием контекстного управления
        with Image.open(image_path) as image:
            # Добавляем изображение в список
            images.append(img_to_array(image))

            # Определяем метку класса и добавляем в список меток
            if folder_name == "6":
                label = 1
            else:
                label = 0
            print(f"Image: {file_name}, Folder: {folder_name}, Label: {label}")

            labels.append(label)


        # Если количество изображений достигло размера пакета или это последнее изображение в папке
        if (i + 1) % batch_size == 0 or (i + 1) == len(os.listdir(folder_path)):
            # Преобразование списка изображений в массив numpy
            images_array = np.array(images)

            # Нормализация значений пикселей
            images_normalized = images_array / 255.0

            # Преобразование меток классов в бинарный формат
            label_encoder = LabelEncoder()
            labels_encoded = label_encoder.fit_transform(labels)
            labels_categorical = to_categorical(labels_encoded)

            # Сохранение подготовленных данных
            np.save(f"images_batch_{i // batch_size}.npy", images_normalized)
            np.save(f"labels_batch_{i // batch_size}.npy", labels_categorical)

            # Очищаем списки для следующего пакета
            images = []
            labels = []

print("Подготовка данных завершена.")
