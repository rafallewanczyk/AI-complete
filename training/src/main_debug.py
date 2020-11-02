# %%
from src.LSTMmodel import LSTMmodel
from src.tokentest import *

path = 'C:\\Users\\Lewanczyk\\Desktop\\ai-complete\\training\\data\\pysource'
number_of_files = 100
# %%
vocab, rev_vocab = generate_vocabs(range(1, number_of_files), path, 2000)

# %%
model = LSTMmodel(vocab, rev_vocab, window_size=5)

# %%
# input for files : [parse_file(file_index, path) for file_index in range(1, 100)]
# for x in model.generate_batch(
#         [['file2', [1, 2, 3, 4, 5, 6, 7, 8, 9]], ['file2', [10, 20, 30, 40, 50, 60, 70, 80, 90]]]):
#     print('executed')
# print('test')

model.train([parse_file(file_index, path) for file_index in range(1, number_of_files)], 3, 32)