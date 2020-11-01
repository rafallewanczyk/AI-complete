from .model import ModelInterface
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences


class LSTMmodel(ModelInterface):

    def __init__(self, vocab, rev_vocab, window_size):
        super().__init__(vocab, rev_vocab, window_size)

        self.model = Sequential()
        self.model.add(Embedding(len(self.vocab), 50, input_length=window_size))
        self.model.add(LSTM(100, return_sequences=True))
        self.model.add(LSTM(100))
        self.model.add(Dense(100, activation='relu'))
        self.model.add(Dense(len(vocab), activation='softmax'))

        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def summary(self):
        return self.model.summary()

    def save(self):
        pass

    def load(self):
        pass

    def _train_on_batch(self, X, y):
        pass

    def train(self, X, y, epochs, batch_size):
        self.model.fit(X, y, batch_size=batch_size, epochs=epochs)

    def predict(self, seed, n):
        text = []
        for i in range(n):
            encoded = []
            for word in seed:
                encoded.append(self.vocab[word])

            encoded = np.array(encoded)
            encoded = pad_sequences([encoded], maxlen=len(seed), truncating='pre')
            y_predict = self.model.predict_classes(encoded)
            text.append(self.rev_vocab[y_predict[0]])
            seed.append(self.rev_vocab[y_predict[0]])
            seed = seed[-len(seed) + 1:]
        return ' '.join(text)