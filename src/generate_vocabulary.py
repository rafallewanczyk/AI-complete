import src.utils as utils
import json
from tqdm.auto import tqdm
from collections import Counter
import pickle


def generate_vocabulary(files, suffixes, number_of_words, name='.\\vocabulary'):
    print('generating vocabulary')

    print("found %s file matches. " % len(files))

    token_count = Counter()
    files_done = 0
    for file_name in tqdm(files):
        tokens = utils.my_tokenize(file_name)
        for token in tokens:
            if len(token) == 0:
                continue
            try:
                token_count[token] += 1
            except:
                token_count[token] = 1
        files_done += 1

    result = token_count.most_common(number_of_words)
    under_threshold = token_count.most_common(number_of_words + 10)
    print('threshold:', result[-1][1])
    print('last 10 words under threshold', [x[0] for x in under_threshold[-10:]])
    with open(name, 'wb') as f:
        pickle.dump(result, f)
    return result


# generate_vocabulary(['..\\data'], ['py'], 70000)
