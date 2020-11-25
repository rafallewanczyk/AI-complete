from collections import Counter


class Ngram:

    def __init__(self, n, sequence):
        self.n = n
        self.m = n - 1
        self.sequence = sequence
        self.subcounter = Counter()
        self.counter = Counter()
        self._set_probabilities()

    def _set_probabilities(self):
        if self.n <= len(self.sequence):
            for i in range(self.n, len(self.sequence) + 1):
                subsequence = tuple(self.sequence[i - self.n:i])
                self.counter[tuple(subsequence)] += 1

            for i in range(self.m, len(self.sequence)):
                subsequence = tuple(self.sequence[i - self.m:i])
                self.subcounter[tuple(subsequence)] += 1


    def predict(self, seed):
        if len(seed) > self.m:
            seed = seed[-self.m:]
        result = []
        seed = tuple(seed)
        if seed not in self.subcounter:
            return result
        else:
            for seq in self.counter:
                if seq[:-1] == seed:
                    result.append((seq[-1], self.counter[seq] / self.subcounter[seed]))
        return result


# ngram = Ngram(2, [1, 2, 3, 4, 1, 2, 3, 5, 1, 2, 2, 1, 2, 2, 1, 2, 2])
# ngram.set_probabilities()
# print(ngram.predict([1,1,11,2,3]))
