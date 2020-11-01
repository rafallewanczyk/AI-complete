#%%
from src.LSTMmodel import LSTMmodel
from src.tokentest import *

path = 'C:\\Users\\rafal\\OneDrive\\Pulpit\\ai-complete\\training\\data\\pysource'
#%%
vocab, rev_vocab = generate_vocabs(range(1, 3), path, 2000)

#%%
model = LSTMmodel(vocab, rev_vocab, window_size=5)

#%%
#input for files : [parse_file(file_index, path) for file_index in range(1, 100)]
model.generate_batch([['file2', [1,2,3,4,5,6,7,8,9]],['file2', [10,20,30,40,50,60,70,80,90]]])
