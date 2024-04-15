import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

# Путь к сохраненной модели
model_path = "model.keras"

# Загрузка модели
model = load_model(model_path)

# Путь к изображению, которое нужно классифицировать
image_path = "/home/hetsu/Desktop/Neural-networks/lab6/data6/6/1.png"

# Загрузка изображения и предварительная обработка
img = image.load_img(image_path, target_size=(150, 150))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)  # Предварительная обработка для VGG16

# Предсказание класса изображения
prediction = model.predict(img_array)
predicted_class = np.argmax(prediction)

# Вывод результата
if predicted_class == 0:
    print("Изображение принадлежит классу 0")
else:
    print("Изображение принадлежит классу 1")
