import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from src.neural_network import create_neural_network, train_neural_network_1, split_data, Neuron, print_predictions_and_targets

def load_image_data(filename):
    img = Image.open(filename)
    img_array = np.array(img)
    data = img_array.reshape(-1, 3).tolist()
    return data

def add_color_labels(data, threshold=128):
    """Пример функции для разметки данных:
    считаем, что пиксель цветной, если хотя бы один из RGB компонентов больше порога.
    """
    labeled_data = []
    for pixel in data:
        is_color = 1 if any(c > threshold for c in pixel) else 0
        labeled_data.append([pixel, [is_color]])
    return labeled_data

# Загрузка данных из изображения
data = load_image_data("kotik.jpg")

# Добавление целевых значений (пример с порогом)
labeled_data = add_color_labels(data)

# Разделение данных
train_data, test_data = split_data(labeled_data, 0.8)

# Создание сети (1 нейрон, 3 входа - R, G, B)
net = create_neural_network(1, 3)

# Обучение
train_neural_network_1(net, train_data, 100, 0.01, 0.01)
# train_neural_network_2(net, train_data, 100, 0.01, 0.01)

# Тестирование
print_predictions_and_targets(net, test_data)

# ... (анализ результатов и подбор весов) ...
