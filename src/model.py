import pickle
import time

import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dense
from tensorflow.keras.layers import GRU, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.layers import Convolution1D

import src.utils as utils
from src.dataset import Dataset


class Model:

    def __init__(self, vocab_size, embedding_dim, rnn_units, batch_size, win_size, checkpoint_name):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.rnn_units = rnn_units
        self.batch_size = batch_size
        self.win_size = win_size
        self.optimizer = tf.keras.optimizers.Adam()
        self.checkpoint_name = checkpoint_name

    def build_model(self):
        self.model = Sequential()
        self.model.add(Embedding(self.vocab_size, self.embedding_dim, batch_input_shape=[self.batch_size, None]))
        self.model.add(Convolution1D(self.embedding_dim, kernel_size=1, activation='relu'))
        self.model.add(
            LSTM(self.rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'))
        # self.model.add(
        #     GRU(self.rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'))
        self.model.add(Dense(self.vocab_size))

    def compile(self, mode='training'):
        if mode == 'training':
            self.model.compile(optimizer='adam', loss=self.loss)
        if mode == 'predict':
            self.model.build(tf.TensorShape([1, None]))

    def train(self, vocab_path, tokenized_files, epochs):
        vocab = self.read_vocab(vocab_path)

        self.dataset = Dataset(vocab, self.win_size)
        self.build_model()
        print(self.model.summary())
        self.compile('training')

        for epoch in range(epochs):
            start = time.time()
            self.model.reset_states()
            loss = 99999999

            batch_start = time.time()
            for (batch_n, (x, y)) in enumerate(self.dataset.next_batch(tokenized_files, self.batch_size)):
                loss = self.train_step(x, y)
                if batch_n % 500 == 0:
                    with open('.\\checkpoints\\losses.txt', 'a') as f:
                            f.write(loss.numpy().__str__())
                            f.write('\n')
                    print(f'Epoch {epoch} Batch {batch_n} Loss {loss} in {"%2f" %(time.time()-batch_start)}')
                    batch_start = time.time()

                if batch_n % 1000 == 0:
                    self.save(self.checkpoint_name)

            print(f'Epoch {epoch + 1} Loss {loss}')
            print(f'Time taken for 1 epoch {time.time() - start}')
        self.save(self.checkpoint_name)

    def prepare_predictions(self, vocab_path, checkpoint=None):
        vocab = self.read_vocab(vocab_path)
        self.dataset = Dataset(vocab, self.win_size)

        self.batch_size = 1
        self.build_model()
        if checkpoint != None:
            self.load(checkpoint)

        self.compile(mode='predict')
        print(self.model.summary())

    def get_prediction(self, seed_string, number=1):
        input_tokens = self.dataset.token2id(utils.tokenize_string(seed_string))
        if len(input_tokens) == 0:
            return ['<UNK>']
        input_tokens = tf.expand_dims(input_tokens, 0)

        predictions = []
        predicted_ids =[]
        text = []
        temperature = 1.0

        self.model.reset_states()
        for i in range(number):
            predictions = self.predict(input_tokens)
            # remove the batch dimension
            predictions = tf.squeeze(predictions, 0)
            # using a categorical distribution to predict the character returned by the model
            predictions = predictions / temperature
            predicted_id = tf.math.argmax(predictions[-1]).numpy()
            predicted_ids = tf.math.top_k(predictions[-1], k=5)
            # predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

            # Pass the predicted character as the next input to the model
            # along with the previous hidden state
            input_tokens= tf.expand_dims([predicted_id], 0)

            # text.append(self.dataset.id2token([predicted_id])[0])
        # return ' '.join(text)
        return self.dataset.id2token(predicted_ids[1].numpy())



    def predict(self, x):
        return self.model(x)

    def train_step(self, X, y):
        with tf.GradientTape() as tape:
            predictions = self.predict(X)
            loss = tf.reduce_mean(self.loss(y, predictions))
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            return loss

    def loss(self, labels, logits):
        return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

    def read_vocab(self, vocab_path):
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        return vocab



    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

#
# vocab_size = 10001
# embedding_dim = 128
# rnn_units = 512
# batch_size = 128
# win_size = 10
# model = Model(vocab_size, embedding_dim, rnn_units, batch_size, win_size)
# # model.train('.\\vocabulary', '..\\data\\django\\django\\apps', 4)
# model.prepare_predictions('.\\vocab10k', '..\\checkpoints\\10k.h5')
# print(model.get_prediction('import functools', 100))
