def base_model(num_epochs, training_reviews, test_reviews, training_stars, test_stars):
    import tensorflow as tf
    import numpy as np
    vocab_size = 5000
    embedding_dim = 64
    max_length = 200
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = '<OOV>'
    filters = 250
    kernel_size = 3
    hidden_dims = 250
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words = vocab_size, oov_token = oov_tok)
    tokenizer.fit_on_texts(training_reviews)
    training_sequences = tokenizer.texts_to_sequences(training_reviews)
    training_padded = tf.keras.preprocessing.sequence.pad_sequences(training_sequences, maxlen = max_length, padding = padding_type, truncating = trunc_type)
    test_sequences = tokenizer.texts_to_sequences(test_reviews)
    test_padded = tf.keras.preprocessing.sequence.pad_sequences(test_sequences, maxlen = max_length, padding = padding_type, truncating = trunc_type)
    training_stars = np.array([int(i) for i in training_stars])
    test_stars = np.array([int(i) for i in test_stars])
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length = max_length))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)))
    model.add(tf.keras.layers.Dense(embedding_dim, activation='elu'))
    model.add(tf.keras.layers.Dense(6, activation='softmax'))
    model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    history = model.fit(training_padded, training_stars, epochs = num_epochs, validation_data = (test_padded, test_stars), verbose=2)
    return model, history

def gru_model (num_epochs, training_reviews, test_reviews, training_stars, test_stars):
    import tensorflow as tf
    import numpy as np
    vocab_size = 5000
    embedding_dim = 200
    max_length = 200
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = '<OOV>'
    filters = 250
    kernel_size = 3
    hidden_dims = 250
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words = vocab_size, oov_token = oov_tok)
    tokenizer.fit_on_texts(training_reviews)
    embeddings = get_embeddings('glove.twitter.27B.200d.txt', tokenizer.word_index, vocab_size, embedding_dim)
    training_sequences = tokenizer.texts_to_sequences(training_reviews)
    training_padded = tf.keras.preprocessing.sequence.pad_sequences(training_sequences, maxlen = max_length, padding = padding_type, truncating = trunc_type)
    test_sequences = tokenizer.texts_to_sequences(test_reviews)
    test_padded = tf.keras.preprocessing.sequence.pad_sequences(test_sequences, maxlen = max_length, padding = padding_type, truncating = trunc_type)
    training_stars = np.array([int(i) for i in training_stars])
    test_stars = np.array([int(i) for i in test_stars])
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(vocab_size, embedding_dim, embeddings_initializer = tf.keras.initializers.Constant(embeddings), input_length = max_length))
    model.add(tf.keras.layers.SpatialDropout1D(.5))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim, return_sequences = True)))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(embedding_dim, return_sequences = True)))
    model.add(tf.keras.layers.GlobalMaxPooling1D())
    model.add(tf.keras.layers.Dense(6, activation='softmax'))
    model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    history = model.fit(training_padded, training_stars, epochs = num_epochs, validation_data = (test_padded, test_stars), verbose=2)
    return model, history
def test_on_random_subset(num, reviews, stars, training_reviews, vocab_size, max_length, model):
    import random
    import numpy as np
    import tensorflow as tf
    rand_start = random.randint(0, len(reviews) - num)
    reviews = reviews[rand_start:rand_start + num]
    stars = stars[rand_start:rand_start + num]
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words = vocab_size, oov_token = '<OOV>')
    tokenizer.fit_on_texts(training_reviews)
    test_sequences = tokenizer.texts_to_sequences(reviews)
    test_padded = tf.keras.preprocessing.sequence.pad_sequences(test_sequences, maxlen = max_length, padding = 'post', truncating = 'post')

    ret = model.evaluate(test_padded, stars, verbose = 0)
    return ret[1]

def get_embeddings(file, word_index, max_features, embed_size):
    import numpy as np
    def get_coefs(word, *arr): 
        return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(file))
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.zeros((nb_words, embed_size))
    missed = []
    for word, i in word_index.items():
        if i >= max_features: break
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            missed.append(word)
    return embedding_matrix

def kaggle_model(num_epochs, tokenizer, training_reviews, test_reviews, training_stars, test_stars):
    import tensorflow as tf
    import numpy as np
    K = tf.keras.backend
    vocab_size = 5000
    embedding_dim = 200
    max_length = 250
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = '<OOV>'
    inp = tf.keras.Input(shape = (1, max_length))
    x = tf.keras.layers.Flatten()(inp)
    x = tf.keras.layers.Embedding(vocab_size, embedding_dim, embeddings_initializer = tf.keras.initializers.Constant(embeddings), trainable = True)(x)
    x = tf.keras.layers.SpatialDropout1D(0.5)(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(40, return_sequences = True))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(40, return_sequences = True))(x)
    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(x)
    max_pool = tf.keras.layers.GlobalMaxPooling1D()(x)
    conc = tf.keras.layers.concatenate([avg_pool, max_pool])
    outp = tf.keras.layers.Dense(1, activation = 'relu')(conc)"""
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    history = model.fit(training_padded, training_stars, batch_size = 64, epochs = num_epochs, validation_data = (test_padded, test_stars), verbose=2)
    return model, history

def single_tree_node(vocab_size, embedding_dim, embeddings):
    import tensorflow as tf
    import numpy as np
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Embedding(vocab_size, embedding_dim, embeddings_initializer = tf.keras.initializers.Constant(embeddings), trainable = True))
    model.add(tf.keras.layers.SpatialDropout1D(0.5))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(40, return_sequences = True)))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(40, return_sequences = True)))
    model.add(tf.keras.layers.GlobalAveragePooling1D())
    model.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    return model

def tree_model(num_epochs, training_reviews, test_reviews, training_stars, test_stars):
    import tensorflow as tf
    import numpy as np
    vocab_size = 5000
    embedding_dim = 200
    max_length = 200
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = '<OOV>'
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words = vocab_size, oov_token = oov_tok)
    tokenizer.fit_on_texts(training_reviews)
    embeddings = get_embeddings('glove.twitter.27B.200d.txt', tokenizer.word_index, vocab_size, embedding_dim)
    training_sequences = tokenizer.texts_to_sequences(training_reviews)
    training_padded = tf.keras.preprocessing.sequence.pad_sequences(training_sequences, maxlen = max_length, padding = padding_type, truncating = trunc_type)
    test_sequences = tokenizer.texts_to_sequences(test_reviews)
    test_padded = tf.keras.preprocessing.sequence.pad_sequences(test_sequences, maxlen = max_length, padding = padding_type, truncating = trunc_type)
    training_padded = np.array([np.array([i]) for i in training_padded])
    test_padded = np.array([np.array([i]) for i in test_padded])
    l1 = single_tree_node(vocab_size, embedding_dim, embeddings) 
    l21 = single_tree_node(vocab_size, embedding_dim, embeddings) 
    l22 = single_tree_node(vocab_size, embedding_dim, embeddings) 
    l31 = single_tree_node(vocab_size, embedding_dim, embeddings) 

    l1.fit(training_padded, np.array([np.array([1]) if i[0] == 1 or i[1] == 1 or i[2] == 1 else np.array([0]) for i in training_stars]), batch_size = 64, epochs = num_epochs, validation_data = (test_padded, np.array([np.array([1]) if i[0] == 1 or i[1] == 1 or i[2] == 1 else np.array([0]) for i in test_stars])), verbose=2)
    l21.fit(training_padded, np.array([np.array([1]) if i[0] == 1 or i[1] == 1 else np.array([0]) for i in training_stars]), batch_size = 64, epochs = num_epochs, validation_data = (test_padded, np.array([np.array([1]) if i[0] == 1 or i[1] == 1 else np.array([0]) for i in test_stars])), verbose=2)
    l22.fit(training_padded, np.array([np.array([1]) if i[3] == 1 else np.array([0]) for i in training_stars]), batch_size = 64, epochs = num_epochs, validation_data = (test_padded, np.array([np.array([1]) if i[3] == 1 else np.array([0]) for i in test_stars])), verbose=2)
    l31.fit(training_padded, np.array([np.array([1]) if i[0] == 1 else np.array([0]) for i in training_stars]), batch_size = 64, epochs = num_epochs, validation_data = (test_padded, np.array([np.array([1]) if i[0] == 1 else np.array([0]) for i in test_stars])), verbose=2)
    return l1, l21, l22, l31, tokenizer

def construct_tree_model(l1, l21, l22, l31):
    def model(review):
        l1_out = l1.predict(review)
        if l1_out[0][0] < .5:
            l22_out = l22.predict(review)
            if l22_out[0][0] < .5:
                return 5
            else:
                return 4
        else:
            l21_out = l21.predict(review)
            if l21_out[0][0] < .5:
                return 3
            else:
                l31_out = l31.predict(review)
                if l31_out[0][0] < .5:
                    return 2
                else:
                    return 1
    return model

def create_padded_from_tokenizer(tokenizer, reviews):
    import tensorflow as tf
    import numpy as np
    max_length = 200
    trunc_type = 'post'
    padding_type = 'post'
    sequences = tokenizer.texts_to_sequences(reviews)
    padded = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen = max_length, padding = padding_type, truncating = trunc_type)
    return np.array([np.array([i]) for i in padded])

def flat_model(num_epochs, training_reviews, test_reviews, training_stars, test_stars):
    import tensorflow as tf
    import numpy as np
    vocab_size = 5000
    embedding_dim = 200
    max_length = 200
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = '<OOV>'
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words = vocab_size, oov_token = oov_tok)
    tokenizer.fit_on_texts(training_reviews)
    embeddings = get_embeddings('glove.twitter.27B.200d.txt', tokenizer.word_index, vocab_size, embedding_dim)
    training_sequences = tokenizer.texts_to_sequences(training_reviews)
    training_padded = tf.keras.preprocessing.sequence.pad_sequences(training_sequences, maxlen = max_length, padding = padding_type, truncating = trunc_type)
    test_sequences = tokenizer.texts_to_sequences(test_reviews)
    test_padded = tf.keras.preprocessing.sequence.pad_sequences(test_sequences, maxlen = max_length, padding = padding_type, truncating = trunc_type)
    training_padded = np.array([np.array([i]) for i in training_padded])
    test_padded = np.array([np.array([i]) for i in test_padded])
    m1 = single_tree_node(vocab_size, embedding_dim, embeddings)
    m2 = single_tree_node(vocab_size, embedding_dim, embeddings)
    m3 = single_tree_node(vocab_size, embedding_dim, embeddings)
    m4 = single_tree_node(vocab_size, embedding_dim, embeddings)
    m5 = single_tree_node(vocab_size, embedding_dim, embeddings)

    m1.fit(training_padded, np.array([np.array([1]) if i[0] == 1 else np.array([0]) for i in training_stars]), batch_size = 64, epochs = num_epochs, validation_data = (test_padded, np.array([np.array([1]) if i[0] == 1 else np.array([0]) for i in test_stars])), verbose=2)
    m2.fit(training_padded, np.array([np.array([1]) if i[1] == 1 else np.array([0]) for i in training_stars]), batch_size = 64, epochs = num_epochs, validation_data = (test_padded, np.array([np.array([1]) if i[1] == 1 else np.array([0]) for i in test_stars])), verbose=2)
    m3.fit(training_padded, np.array([np.array([1]) if i[2] == 1 else np.array([0]) for i in training_stars]), batch_size = 64, epochs = num_epochs, validation_data = (test_padded, np.array([np.array([1]) if i[2] == 1 else np.array([0]) for i in test_stars])), verbose=2)
    m4.fit(training_padded, np.array([np.array([1]) if i[3] == 1 else np.array([0]) for i in training_stars]), batch_size = 64, epochs = num_epochs, validation_data = (test_padded, np.array([np.array([1]) if i[3] == 1 else np.array([0]) for i in test_stars])), verbose=2)
    m5.fit(training_padded, np.array([np.array([1]) if i[4] == 1 else np.array([0]) for i in training_stars]), batch_size = 64, epochs = num_epochs, validation_data = (test_padded, np.array([np.array([1]) if i[4] == 1 else np.array([0]) for i in test_stars])), verbose=2)
    return m1, m2, m3, m4, m5

def tokenizer_sorted_out(texts):
    import tensorflow as tf
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words = 5000, oov_token = '<OOV>')
    tokenizer.fit_on_texts(texts)
    counts = tokenizer.word_counts
    sorted_thing = []
    for i in counts.keys():
        sorted_thing.append((i, counts[i]))
    sorted_thing.sort(key = lambda x: x[1])
    sorted_thing = sorted_thing[::-1]
    return sorted_thing

def most_unique_words(s1, s2):
    s1_sum = sum([i[1] for i in s1])
    s2_sum = sum([i[1] for i in s2])
    s1 = [(i[0], i[1] / s1_sum) for i in s1]
    s2 = [(i[0], i[1] / s2_sum) for i in s2]
    s1 = dict(s1)
    s2 = dict(s2)
    s1_unique = []
    s2_unique = []
    crossover = []
    for i in s1.keys():
        if i in s2:
            crossover.append((i, s1[i] - s2[i]))
        else:
            s1_unique.append(i)
    for i in s2.keys():
        if i in s1:
            continue
        else:
            s2_unique.append(i)
    crossover.sort(key = lambda x: x[1])
    return s1_unique, s2_unique, crossover

def baseline(num_epochs, tokenizer, training_reviews, test_reviews, training_stars, test_stars):
    import tensorflow as tf
    import numpy as np
    training_padded = tf.keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences(training_reviews), maxlen = 250, padding = 'post', truncating = 'post')
    test_padded = tf.keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences(test_reviews), maxlen = 250, padding = 'post', truncating = 'post')
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(5000, 250))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(250)))
    model.add(tf.keras.layers.Dense(5, activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    history = model.fit(training_padded, training_stars, batch_size = 32, epochs = num_epochs, validation_data = (test_padded, test_stars), verbose=2)
    return model, history
def baseline_with_trained_embeddings(num_epochs, tokenizer, training_reviews, test_reviews, training_stars, test_stars):
    import tensorflow as tf
    import numpy as np
    embeddings = get_embeddings('glove.twitter.27B.200d.txt', tokenizer.word_index, 5000, 200)
    training_padded = tf.keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences(training_reviews), maxlen = 250, padding = 'post', truncating = 'post')
    test_padded = tf.keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences(test_reviews), maxlen = 250, padding = 'post', truncating = 'post')
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(5000, 200, embeddings_initializer = tf.keras.initializers.Constant(embeddings), trainable = True))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(200)))
    model.add(tf.keras.layers.Dense(5, activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    history = model.fit(training_padded, training_stars, batch_size = 32, epochs = num_epochs, validation_data = (test_padded, test_stars), verbose=2)
    return model, history

def baseline_with_custom_metrics(num_epochs, tokenizer, training_reviews, test_reviews, training_stars, test_stars):
    import tensorflow as tf
    import numpy as np
    K = tf.keras.backend
    def custom_accuracy(y_true, y_pred):
        return K.sum(K.cast(K.equal(K.argmax(y_true), K.argmax(y_pred)), dtype = tf.int32)) / 32
    def distance(y_true, y_pred):
        total_corr = K.sum(K.cast(K.equal(K.argmax(y_true), K.argmax(y_pred)), dtype = tf.int32))
        return K.sum(K.cast(K.abs(K.argmax(y_true) - K.argmax(y_pred)), dtype = tf.int32)) / (32 - total_corr)
    def sum_metrics(y_true, y_pred):
        acc = (32 - K.sum(K.cast(K.equal(K.argmax(y_true), K.argmax(y_pred)), dtype = tf.int32))) / 32
        total_corr = K.sum(K.cast(K.equal(K.argmax(y_true), K.argmax(y_pred)), dtype = tf.int32))
        return acc + K.sum(K.cast(K.abs(K.argmax(y_true) - K.argmax(y_pred)), dtype = tf.int32)) / (32 - total_corr)
    embeddings = get_embeddings('glove.twitter.27B.200d.txt', tokenizer.word_index, 5000, 200)
    training_padded = tf.keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences(training_reviews), maxlen = 250, padding = 'post', truncating = 'pre')
    test_padded = tf.keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences(test_reviews), maxlen = 250, padding = 'post', truncating = 'pre')
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(5000, 200, embeddings_initializer = tf.keras.initializers.Constant(embeddings), trainable = True))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(50)))
    model.add(tf.keras.layers.Dense(5, activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy', custom_accuracy, distance, sum_metrics])
    history = model.fit(training_padded, training_stars, batch_size = 32, epochs = num_epochs, validation_data = (test_padded, test_stars), verbose=2)
    return model, history

def tree_node(num_epochs, tokenizer, training_reviews, test_reviews, training_stars, test_stars):
    import tensorflow as tf
    import numpy as np
    K = tf.keras.backend
    def custom_accuracy(y_true, y_pred):
        return K.sum(K.cast(K.equal(K.argmax(y_true), K.argmax(y_pred)), dtype = tf.int32)) / 32
    embeddings = get_embeddings('glove.twitter.27B.200d.txt', tokenizer.word_index, 5000, 200)
    training_padded = tf.keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences(training_reviews), maxlen = 250, padding = 'post', truncating = 'post')
    test_padded = tf.keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences(test_reviews), maxlen = 250, padding = 'post', truncating = 'post')
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(5000, 200, embeddings_initializer = tf.keras.initializers.Constant(embeddings), trainable = True))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(50)))
    model.add(tf.keras.layers.Dense(2, activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy', custom_accuracy])
    history = model.fit(training_padded, training_stars, batch_size = 32, epochs = num_epochs, validation_data = (test_padded, test_stars), verbose=2)
    return model, history

def get_most_confused(pred, true, num):
    import numpy as np
    confused = dict()
    counts = dict()
    for i in range(num):
        counts[str(i)] = 0
        for j in range(num):
            if i != j:
                confused[str(i) + "-" + str(j)] = (0, 0)
    for i in range(len(pred)):
        pred_i = np.argmax(pred[i])
        true_i = np.argmax(true[i])
        counts[str(true_i)] = counts[str(true_i)] + 1
        if true_i == pred_i:
            continue
        else:
            confused[str(true_i) + "-" + str(pred_i)] = (confused[str(true_i) + "-" + str(pred_i)][0] + 1, confused[str(true_i) + "-" + str(pred_i)][1] + pred[i][pred_i])
    for i in confused.keys():
        if confused[i][0] != 0:
            confused[i] = (confused[i][0], confused[i][1] / confused[i][0])
        else:
            confused[i] = (0, 0)
    return confused, counts

def get_probs_when(pred, true, ind):
    import numpy as np
    prob_to_perc = dict()
    steps = [.1, .2, .3, .4, .5, .6, .7, .8, .9]
    for i in steps:
        options = []
        corr = 0
        for j in range(len(pred)):
            if np.argmax(pred[j]) == ind and pred[j][ind] >= i:
                options.append(j)
        for j in options:
            if np.argmax(pred[j]) == np.argmax(true[j]):
                corr += 1
        if len(options) != 0:
            prob_to_perc[str(i)] = corr / len(options)
        else:
            prob_to_perc[str(i)] = 0
    return prob_to_perc

def get_binaries_sets(train_df, test_df, num, ind1, ind2, segmentation = .8):
    import numpy as np
    training_reviews, training_stars, test_reviews, test_stars = get_fit_samples(train_df, test_df, num * 10, segmentation)
    ret_tr = [training_reviews[i] for i in range(len(training_reviews)) if np.argmax(training_stars[i]) == ind1][:int(num * segmentation / 2)] + [training_reviews[i] for i in range(len(training_reviews)) if np.argmax(training_stars[i]) == ind2][:int(num * segmentation / 2)]
    ret_ts = np.array([training_stars[i] for i in range(len(training_reviews)) if np.argmax(training_stars[i]) == ind1][:int(num * segmentation / 2)] + [training_stars[i] for i in range(len(training_reviews)) if np.argmax(training_stars[i]) == ind2][:int(num * segmentation / 2)])
    ret_ter = [test_reviews[i] for i in range(len(test_reviews)) if np.argmax(test_stars[i]) == ind1][:int(num * (1 - segmentation) / 2)] + [test_reviews[i] for i in range(len(test_reviews)) if np.argmax(test_stars[i]) == ind2][:int(num * (1 - segmentation) / 2)]
    ret_tes = np.array([test_stars[i] for i in range(len(test_reviews)) if np.argmax(test_stars[i]) == ind1][:int(num * (1 - segmentation) / 2)] + [test_stars[i] for i in range(len(test_reviews)) if np.argmax(test_stars[i]) == ind2][:int(num * (1 - segmentation) / 2)])
    return ret_tr, ret_ts, ret_ter, ret_tes

def model_to_func(model):
    def out(inp):
        return model.predict(inp)
    return out

def helper_binary_to_func(model, rep1, rep2):
    import numpy as np
    def out(inp):
        ret = []
        pred = model.predict(inp)
        for i in pred:
            if np.argmax(i) == 0:
                ret.append(rep1)
            else:
                ret.append(rep2)
        return np.array(ret)
    return out

def stack_helper(model, helper, guess, confidence):
    import numpy as np
    def out(inp):
        inter = model(inp)
        helper_guesses = []
        for i in range(len(inp)):
            if np.argmax(inter[i]) == guess and np.max(inter[i]) <= confidence:
                helper_guesses.append(helper(np.array([inter[i]]))[0])
            else:
                helper_guesses.append(inter[i])
        return np.array(helper_guesses)
    return out

def baseline_with_custom_metrics_dropout(num_epochs, tokenizer, training_reviews, test_reviews, training_stars, test_stars):
    import tensorflow as tf
    import numpy as np
    K = tf.keras.backend
    def custom_accuracy(y_true, y_pred):
        return K.sum(K.cast(K.equal(K.argmax(y_true), K.argmax(y_pred)), dtype = tf.int32)) / 32
    def distance(y_true, y_pred):
        total_corr = K.sum(K.cast(K.equal(K.argmax(y_true), K.argmax(y_pred)), dtype = tf.int32))
        return K.sum(K.cast(K.abs(K.argmax(y_true) - K.argmax(y_pred)), dtype = tf.int32)) / (32 - total_corr)
    def sum_metrics(y_true, y_pred):
        acc = (32 - K.sum(K.cast(K.equal(K.argmax(y_true), K.argmax(y_pred)), dtype = tf.int32))) / 32
        total_corr = K.sum(K.cast(K.equal(K.argmax(y_true), K.argmax(y_pred)), dtype = tf.int32))
        return acc + K.sum(K.cast(K.abs(K.argmax(y_true) - K.argmax(y_pred)), dtype = tf.int32)) / (32 - total_corr)
    embeddings = get_embeddings('glove.twitter.27B.200d.txt', tokenizer.word_index, 5000, 200)
    training_padded = tf.keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences(training_reviews), maxlen = 250, padding = 'post', truncating = 'post')
    test_padded = tf.keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences(test_reviews), maxlen = 250, padding = 'post', truncating = 'post')
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(5000, 200, embeddings_initializer = tf.keras.initializers.Constant(embeddings), trainable = True))
    model.add(tf.keras.layers.SpatialDropout1D(0.5))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(50)))
    model.add(tf.keras.layers.Dense(5, activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy', custom_accuracy, distance, sum_metrics])
    history = model.fit(training_padded, training_stars, batch_size = 32, epochs = num_epochs, validation_data = (test_padded, test_stars), verbose=2)
    return model, history
def stars_to_probs(stars, mid_val):
    import numpy as np
    stars_out = []
    for i in stars:
        ind = np.argmax(i)
        app = [0, 0, 0, 0, 0]
        if ind == 0:
            app[0] = mid_val
            app[1] = (1 - mid_val) / 2
            app[2] = (1 - mid_val) / 2
        if ind == 1:
            app[0] = (1 - mid_val) / 2
            app[1] = mid_val
            app[2] = (1 - mid_val) / 2
        if ind == 2:
            app[1] = (1 - mid_val) / 2
            app[2] = mid_val
            app[3] = (1 - mid_val) / 2
        if ind == 3:
            app[2] = (1 - mid_val) / 2
            app[3] = mid_val
            app[4] = (1 - mid_val) / 2
        if ind == 4:
            app[2] = (1 - mid_val) / 2
            app[3] = (1 - mid_val) / 2
            app[4] = mid_val
        stars_out.append(np.array(app))
    return np.array(stars_out)
EOF
