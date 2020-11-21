import numpy as np


class Dataset:
    def __init__(self, keywords, win_size):
        self.keywords = keywords
        self.win_size = win_size
        self.word_to_id = {'<UNK>': 0}
        self.id_to_word = {0: '<UNK>'}
        for i, k in enumerate(self.keywords, 1):
            self.word_to_id[k[0]] = i
            self.id_to_word[i] = k[0]

    def token2id(self, tokens):
        ids = []
        for token in tokens:
            if token not in self.word_to_id:
                ids.append(self.word_to_id['<UNK>'])
            else:
                ids.append(self.word_to_id[token])
        return ids

    def id2token(self, ids):
        tokens = []
        for i in ids:
            if i not in self.id_to_word:
                tokens.append(self.id_to_word[0])
            else:
                tokens.append(self.id_to_word[i])
        return tokens

    def next_batch(self, filedata, batch_size=1):
        tokenids = [(name, self.token2id(tokens))
                    for name, tokens in filedata]
        # filter out files with < win_size+1 tokens
        filtered = [tokids for _, tokids in tokenids
                    if len(tokids) >= self.win_size + 1]
        weights = [len(toks) - self.win_size for toks in filtered]
        num_wins = sum(weights)
        print(f'number of batches : {num_wins // batch_size}')
        for i in range(0, num_wins, batch_size):
            # Pick the file by weights
            idxs = [np.random.randint(len(weights)) for _ in range(batch_size)]
            # Beginning Index of the window
            widxs = [np.random.randint(len(filtered[idx]) - self.win_size)
                     for idx in idxs]
            Xs = [filtered[idx][widx:widx + self.win_size]
                  for widx, idx in zip(widxs, idxs)]
            ys = [filtered[idx][widx + 1:widx + self.win_size + 1]
                  for widx, idx in zip(widxs, idxs)]
            if Xs and ys:
                yield np.array(Xs), np.array(ys)
            else:
                continue
