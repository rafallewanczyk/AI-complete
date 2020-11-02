import numpy as np
from tqdm import tqdm


class ModelInterface:

    def __init__(self, vocab, rev_vocab, window_size):
        """
        :param vocab: vocabulary word -> vector
        :param rev_vocab: vocabulary vector -> word
        """
        self.vocab = vocab
        self.rev_vocab = rev_vocab
        self.window_size = window_size

    def save(self):
        """
        saves trained nn to file
        """
        raise NotImplementedError("Implement me!!")

    def load(self):
        """
        loads trained nn from file
        """
        raise NotImplementedError("Implement me!!")

    def token_to_int(self, token):
        if token in self.vocab:
            return self.vocab[token]
        else:
            return self.vocab['UNKNOWN']

    def train(self, parsed_files, epochs, batch_size):
        """
        generates batches and performs training on them
        :param parsed_files: training set
        :param epochs: number of epochs
        :param batch_size: size of batch required for _train_on_batch
        :return: none
        """

        # convert words to integers:
        parsed_files = [(name, [self.token_to_int(token) for token in token_list]) for name, token_list in parsed_files]
        smooth_loss = None
        smooth_acc = None

        def smooth_update(sx, x):
            if sx is None:
                return x
            else:
                return 0.99 * sx + 0.01 * x

        for epoch in range(epochs):
            generator = self.generate_batch(parsed_files, 32)
            windows_number = sum([len(tokens) for _, tokens in parsed_files]) - len(parsed_files) * self.window_size

            progress = tqdm(range(0, windows_number, batch_size))
            for i in progress:
                X, y = next(generator)
                loss, acc = self._train_on_batch(X, y)

                smooth_loss = smooth_update(smooth_loss, loss)
                smooth_acc = smooth_update(smooth_acc, acc)

                if i % (windows_number // 1000) == 0:
                    progress.set_postfix_str(f'loss:{smooth_loss}, acc:{smooth_acc}')

    def _train_on_batch(self, Xs, ys):
        """
        performs training on single batch
        :param Xs: subset of X
        :param ys: subset of y
        :return: none
        """
        raise NotImplementedError("Implement me!!")

    def generate_batch(self, parsed_files, batch_size=32):
        filtered = []
        for name, tokens in parsed_files:
            if len(tokens) >= self.window_size:
                filtered.append(tokens)

        windows_number = sum([len(file) for file in filtered]) - len(filtered) * self.window_size

        for i in range(0, windows_number, batch_size):
            files_indexes = [np.random.randint(0, len(filtered)) for i in range(batch_size)]
            start_indexes = [np.random.randint(len(filtered[idx]) - self.window_size) for idx in files_indexes]

            Xs = [filtered[files_indexes][start_indexes:start_indexes + self.window_size] for
                  files_indexes, start_indexes in zip(files_indexes, start_indexes)]
            ys = [filtered[files_indexes][start_indexes + self.window_size] for files_indexes, start_indexes in
                  zip(files_indexes, start_indexes)]

            yield Xs, ys

    def predict(self, seed, n):
        """
        predicts top n values
        :param n: number of predicted values
        :param seed: input list of tokens
        :return: predicted string
        """
        raise NotImplementedError("Implement me!!")
