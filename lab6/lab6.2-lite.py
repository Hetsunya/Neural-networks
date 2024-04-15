from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras.models import Sequential
from tensorflow.keras.layers import Input
from keras.layers import Flatten, Dense, Dropout
from keras.optimizers import Adam

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
#
# # Загружаем предобученную модель VGG16 без полносвязных слоев
# base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

# Слой ввода с определенной формой
input_layer = Input(shape=(150, 150, 3))

# Загрузка базовой модели VGG16
base_model = VGG16(weights='imagenet', include_top=False, input_tensor=input_layer)


# Создаем новую модель на основе VGG16
model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(6, activation='softmax'))

# Замораживаем веса базовой модели
base_model.trainable = False

# Компиляция модели
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])


# Обучение модели
history = model.fit(
    train_generator,
    epochs=2,
    validation_data=validation_generator
)

# Оценка модели
loss, accuracy = model.evaluate(validation_generator)
print("Потери:", loss)
print("Точность:", accuracy)

# Сохранение модели
# model.save("model.h5")


model.save("model.keras")
