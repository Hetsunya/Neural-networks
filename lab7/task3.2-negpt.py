import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense

# Загрузка данных
verses_df = pd.read_csv("verses.csv")
sentences_df = pd.read_csv("sentences.csv")

# Объединение данных
data_df = pd.concat([verses_df[["russian", "english"]], sentences_df[["russian", "english"]]], ignore_index=True)

# Предварительная обработка
rus_texts = data_df["russian"].tolist()
eng_texts = data_df["english"].tolist()

# Токенизация и создание словарей
rus_tokenizer = Tokenizer()
rus_tokenizer.fit_on_texts(rus_texts)
eng_tokenizer = Tokenizer()
eng_tokenizer.fit_on_texts(eng_texts)

# Преобразование текстов в последовательности индексов
rus_sequences = rus_tokenizer.texts_to_sequences(rus_texts)
eng_sequences = eng_tokenizer.texts_to_sequences(eng_texts)

# Паддинг последовательностей (только для входных данных)
max_len = max(len(s) for s in rus_sequences)
rus_padded = pad_sequences(rus_sequences, maxlen=max_len, padding='post')

# Создание biRNN модели
model = Sequential()
model.add(Embedding(len(rus_tokenizer.word_index) + 1, 256))
model.add(Bidirectional(LSTM(256, return_sequences=True)))
model.add(Bidirectional(LSTM(256)))
model.add(Dense(len(eng_tokenizer.word_index) + 1, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Обучение модели
model.fit(rus_padded, eng_sequences, epochs=10, batch_size=64)

# Пример использования (перевод предложения)
sentence = "Я люблю читать книги."
sequence = rus_tokenizer.texts_to_sequences([sentence])[0]
padded_sequence = pad_sequences([sequence], maxlen=max_len, padding='post')
predicted_probs = model.predict(padded_sequence)[0]
predicted_indices = np.argmax(predicted_probs, axis=1)
predicted_text = eng_tokenizer.sequences_to_texts([predicted_indices])[0]

print("Русский:", sentence)
print("Английский:", predicted_text)