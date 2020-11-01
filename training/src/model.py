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

    def train(self, X, y, epochs, batch_size):
        """
        generates batches and performs training on them
        :param X: training set
        :param y: labels
        :param epochs: number of epochs
        :param batch_size: size of batch required for _train_on_batch
        :return: none
        """
        raise NotImplementedError("Implement me!!")

    def _train_on_batch(self, Xs, ys):
        """
        performs training on single batch
        :param Xs: subset of X
        :param ys: subset of y
        :return: none
        """
        raise NotImplementedError("Implement me!!")

    def generate_batch(self, parsed_files, batch_size = 32):
        filtered = []
        for name, tokens in parsed_files:
            if len(tokens) >= self.window_size:
                filtered.append(tokens)







    def predict(self, seed, n):
        """
        predicts top n values
        :param n: number of predicted values
        :param seed: input list of tokens
        :return: predicted string
        """
        raise NotImplementedError("Implement me!!")
