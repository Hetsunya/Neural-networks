import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from scipy import sparse

# Загрузка данных
data = pd.read_csv("sentences.csv")
X_train_text = np.array(data["russian"])
y_train_text = np.array(data["english"])

# Создание токенизатора для русского языка
tokenizer_ru = Tokenizer()
tokenizer_ru.fit_on_texts(X_train_text)
X_train_seq = tokenizer_ru.texts_to_sequences(X_train_text)

# Преобразование в целочисленные индексы и приведение к одной длине
max_len_ru = max(len(seq) for seq in X_train_seq)
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len_ru, padding='post')

# Создание токенизатора для английского языка
tokenizer_en = Tokenizer()
tokenizer_en.fit_on_texts(y_train_text)
y_train_seq = tokenizer_en.texts_to_sequences(y_train_text)

# Преобразование в целочисленные индексы и приведение к одной длине
max_len_en = max(len(seq) for seq in y_train_seq)
y_train_pad = pad_sequences(y_train_seq, maxlen=max_len_en, padding='post')

# Создание разреженной матрицы для категориальных меток
y_train_sparse = sparse.lil_matrix((len(y_train_pad), len(tokenizer_en.word_index) + 1))
for i, sequence in enumerate(y_train_pad):
    for j, token_index in enumerate(sequence):
        y_train_sparse[i, token_index] = 1

# Создание модели
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=len(tokenizer_ru.word_index) + 1, output_dim=100),
    tf.keras.layers.Bidirectional(tf.keras.layers.SimpleRNN(64)),
    tf.keras.layers.Dense(len(tokenizer_en.word_index) + 1, activation='softmax')
])

# Компиляция модели
model.compile(optimizer='adam', loss='categorical_crossentropy')

# Преобразование разреженной матрицы в массив numpy
y_train_np = y_train_sparse.toarray()

# Обучение модели
model.fit(X_train_pad, y_train_np, epochs=10, batch_size=32)

# Прогнозирование
sample_input = ["Привет, как дела?"]
sample_input_seq = tokenizer_ru.texts_to_sequences(sample_input)
sample_input_pad = pad_sequences(sample_input_seq, maxlen=max_len_ru, padding='post')
predicted_output = model.predict(sample_input_pad)
print("Russian sentence:", sample_input)
print("Predicted English translation:", predicted_output)
