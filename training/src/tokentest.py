from tokenize import tokenize
from io import BytesIO
from collections import Counter
from tqdm import tqdm


def parse_file(file_number, path):
    all_tokens = []
    with open(f'{path}\\{file_number}.py') as f:
        data = f.readlines()
        data = ''.join(data)
        try:
            tokens = tokenize(BytesIO(data.encode('utf-8')).readline)
            all_tokens += [token.string for token in tokens]
        except Exception as e:
            pass
            #todo inform about error occurence
    return file_number, all_tokens


def generate_vocabs(file_indexes, path, vocab_size):
    all_tokens = []
    for i in tqdm(file_indexes):
        name, tokens = parse_file(i, path)
        all_tokens += tokens

    counter = Counter(tokens)

    vocab = {'UNKNOWN': 0}
    reversed_vocab = {0: 'UNKNOWN'}

    for token in enumerate(counter.most_common(vocab_size), 1):
        vocab[token[1][0]] = token[0]
        reversed_vocab[token[0]] = token[1][0]

    return vocab, reversed_vocab


def generate_sequences(tokens, seq_length, vocab):

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

    return sequences

# generate_sequences(['a', 'a', 'b', 'c', 'd', 'b', 'f', 'c', 'c','q','w','e'], 5, 3)
# get_tokens(752, 1)
