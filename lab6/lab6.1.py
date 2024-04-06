import cv2
import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import maximum_filter

# Загружаем изображение
image = cv2.imread("img.jpeg", cv2.IMREAD_GRAYSCALE)

# Расширяем границы изображения нулями
padded_image = np.pad(image, ((1, 1), (1, 1)), mode='constant', constant_values=0)

# Ядро свертки 3x3 заполненное нулями
kernel_zeros = np.array([[0, 0, 0],
                         [0, 1, 0],
                         [0, 0, 0]])

# Ядро свертки 3x3 заполненное единицами
kernel_ones = np.ones((3, 3))

# Произвольное ядро свертки 3x3
kernel_custom = np.array([[0.1, 0.2, 0.3],
                          [0.4, 0.5, 0.6],
                          [0.7, 0.8, 0.9]])

# Применяем свертку
convolved_zeros = convolve2d(padded_image, kernel_zeros, mode='valid')
convolved_ones = convolve2d(padded_image, kernel_ones, mode='valid')
convolved_custom = convolve2d(padded_image, kernel_custom, mode='valid')

# Применяем пулинг с функцией максимума
pooled_max = maximum_filter(image, size=2)

# Выводим результаты
print("Свертка с ядром, заполненным нулями:")
print(convolved_zeros)
print("\nСвертка с ядром, заполненным единицами:")
print(convolved_ones)
print("\nСвертка с произвольным ядром:")
print(convolved_custom)
print("\nПулинг с функцией максимума:")
print(pooled_max)
