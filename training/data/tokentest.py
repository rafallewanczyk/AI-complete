from tokenize import tokenize
from io import BytesIO
from collections import Counter
from tqdm import tqdm


def get_tokens(file_number, path):
    all_tokens = []
    errors = []
    for i in tqdm(range(1, file_number)):
        with open(f'{path}\\{i}.py') as f:
            data = f.readlines()
            data = ''.join(data)
            try:
                tokens = tokenize(BytesIO(data.encode('utf-8')).readline)
                all_tokens += [token.string for token in tokens]
            except Exception as e:
                errors.append(f'error in file {i}: {e}')
                continue

    for error in errors:
        print(error)
    return all_tokens


def generate_sequences(tokens, vocab_size, seq_length):
    counter = Counter(tokens)

    vocab = {'UNKNOWN': 0}
    reversed_vocab = {0: 'UNKNOWN'}

    for token in enumerate(counter.most_common(vocab_size), 1):
        vocab[token[1][0]] = token[0]
        reversed_vocab[token[0]] = token[1][0]

    print(vocab)
    print(reversed_vocab)

    seq_length += 1  # one predicted word
    sequences = []
    for i in tqdm(range(seq_length, len(tokens) + 1)):
        sequence = tokens[i - seq_length:i]
        coded_seq = []
        for token in sequence:
            if token in vocab:
                coded_seq.append(vocab[token])
            else:
                coded_seq.append(vocab['UNKNOWN'])
        sequences.append(coded_seq)

    return vocab, reversed_vocab, sequences

# generate_sequences(['a', 'a', 'b', 'c', 'd', 'b', 'f', 'c', 'c','q','w','e'], 5, 3)
# get_tokens(752, 1)
