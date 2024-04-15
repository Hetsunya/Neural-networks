from keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Загрузка модели
model = load_model("fin.h5")

# Путь к папке с данными
data_dir = "data6"

# Создаем генератор изображений для обучения
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2
)

# Генератор для обучения
train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

# Генератор для валидации
validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

import matplotlib.pyplot as plt
import os

result_dir = "result for test"
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# Предсказание класса для тестовой выборки
predictions = model.predict(validation_generator)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = validation_generator.classes

# Сохранение изображений с подписями
for i in range(len(validation_generator)):
    # Получаем батч изображений и меток из генератора
    batch_images, batch_labels = validation_generator[i]
    # Получаем предсказания модели для текущего батча
    batch_predictions = model.predict(batch_images)
    # Проходим по каждому изображению в батче
    for j in range(len(batch_images)):
        image = batch_images[j]  # Получаем изображение из батча
        true_label = "Ваш вариант" if batch_labels[j][5] == 1 else "Не ваш вариант"  # Получаем метку истинного класса
        predicted_label = "Ваш вариант" if batch_predictions[j][5] == 1 else "Не ваш вариант"  # Получаем предсказанную метку класса
        plt.imshow(image)
        plt.title(f"True: {true_label}, Predicted: {predicted_label}")
        plt.savefig(f"{result_dir}/test_result_{i * len(batch_images) + j}.png")
        plt.close()

print(f"Результаты работы модели сохранены в файлы в папке '{result_dir}'")
