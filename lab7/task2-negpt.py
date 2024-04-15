from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Загрузка текста
text = open("war_and_peace.txt", "r", encoding="utf-8").read().lower()

# Токенизация и создание словаря
num_words = 5
tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts([text])
word_index = tokenizer.word_index

# Подготовка последовательностей
sequences = []
for line in text.split('\n'):
    encoded = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(encoded)):
        sequence = encoded[:i+1]
        sequences.append(sequence)

max_sequence_len = max([len(x) for x in sequences])
sequences = pad_sequences(sequences, maxlen=max_sequence_len, padding='pre')

X, y = sequences[:,:-1], sequences[:,-1]  # y - индексы слов, а не one-hot векторы

y = to_categorical(y, num_classes=len(word_index)+1)

# Создание LSTM модели
model = Sequential()
model.add(Embedding(len(word_index)+1, 128))  # удаляем input_length
model.add(LSTM(128))
model.add(Dense(len(word_index)+1, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# Обучение модели
model.fit(X, y, epochs=5, batch_size=10)

# Генерация текста
seed_text = "Наташа Ростова"
next_words = 10

for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted_probs = model.predict(token_list, verbose=0)[0]
    predicted_index = np.argmax(predicted_probs)
    output_word = tokenizer.index_word[predicted_index]
    seed_text += " " + output_word

print(seed_text)