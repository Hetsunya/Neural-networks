from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

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

# Создание архитектуры нейронной сети
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(6, activation='softmax'))

# Компиляция модели
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Обучение модели
history = model.fit(
    train_generator,
    epochs=3,
    validation_data=validation_generator
)

# Оценка модели
loss, accuracy = model.evaluate(validation_generator)
print("Потери:", loss)
print("Точность:", accuracy)


import matplotlib.pyplot as plt
import os

# Создаем папку для сохранения результатов, если она не существует
result_dir = "result"
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
    batch_predictions = model.predict(  )
    # Проходим по каждому изображению в батче
    for j in range(len(batch_images)):
        image = batch_images[j]  # Получаем изображение из батча
        true_label = "Ваш вариант" if np.argmax(batch_labels[j]) == 1 else "Не ваш вариант"  # Получаем метку истинного класса
        predicted_label = "Ваш вариант" if np.argmax(batch_predictions[j]) == 1 else "Не ваш вариант"  # Получаем предсказанную метку класса
        plt.imshow(image)
        plt.title(f"True: {true_label}, Predicted: {predicted_label}")
        plt.savefig(f"{result_dir}/test_result_{i * len(batch_images) + j}.png")
        plt.close()

print("Результаты работы модели сохранены в файлы в папке 'result'")


