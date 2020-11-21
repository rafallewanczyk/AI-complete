import tokenize
from tqdm.auto import tqdm
import os
import re
from io import BytesIO


def uncomment(fileName, content):
    ## only python ??? 
    if fileName.endswith('.c'):
        return re.sub(r'/\*.*?\*/', '', content,
                      flags=re.MULTILINE | re.DOTALL)
    elif fileName.endswith(('.cpp', '.cc', '.h', '.hh', '.hxx', 'hpp', 'java')):
        pass1 = re.sub(r'/\*.*?\*/', '', content,
                       flags=re.MULTILINE | re.DOTALL)
        pass2 = re.sub(r'//.*', '', pass1)
        return pass2
    elif fileName.endswith('.py'):
        pass1 = re.sub(r'#.*', '', content)
        pass2 = re.sub(r'""".*?"""', '', pass1,
                       flags=re.MULTILINE | re.DOTALL)
        return pass2
    return content


def search_files(data_dirs, suffixes):
    print(data_dirs)
    matches = []  # list of files to read
    for data_dir in data_dirs:
        for root, dirnames, filenames in os.walk(data_dir):
            for filename in filenames:
                if filename.endswith(tuple(suffixes)):
                    matches.append(os.path.join(root, filename))
    return matches


def my_tokenize(fileName, retcontent=False, verbose=False):
    allTokens = []
    try:
        with open(fileName) as f:
            data = f.readlines()
            data = ''.join(data)
            data = uncomment(fileName, data)
            data = [line.strip() + '\n' for line in data.split('\n') if line.rstrip()]
            data = ''.join(data)
            tokens = tokenize.tokenize(BytesIO(data.encode('utf-8')).readline)
            allTokens += [token.string for token in tokens]

    except Exception as e:
        if verbose:
            print(f'skipping file {fileName} due to {e}')
    if not retcontent:
        return allTokens
    else:
        return allTokens, data

def tokenize_string(input_string):
    g = tokenize.tokenize(BytesIO(input_string.encode('utf-8')).readline)
    tokens = []
    try:
        for t in g:
            if t.string != '':
                tokens.append(t.string)
    except tokenize.TokenError:
        pass
    return tokens[1:]

def tokenize_data(files):
    tokenized_files = []
    for file in tqdm(files):
        tokenized_files.append((file, my_tokenize(file)))
    return tokenized_files
