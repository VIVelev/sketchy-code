import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from .config import IMAGE_SIZE, PATH_TO_DATA
from .image import get_preprocessed_img
from .sequence import tokenize_dsl_code

__all__ = [
    'get_sketch_id_code_pair',
    'init_sketch_id_code_map',
    'load_vocabulary',
    'init_word2idx',
    'init_idx2word',
    'data_generator',
]


def get_sketch_id_code_pair(path_to_code):
    sketch_id = path_to_code.split('/')[-1].split('.')[0]
    with open(path_to_code, 'r') as f:
        code = f.read()

    return (sketch_id, code)


def init_sketch_id_code_map(paths):
    sketch_id_code_map = dict()

    for path in paths:
        sketch_id, code = get_sketch_id_code_pair(path)
        sketch_id_code_map[sketch_id] = code

    return sketch_id_code_map


def load_vocabulary(path_to_txt):
    with open(path_to_txt, 'r') as f:
        vocabulary = f.read().split()

    vocabulary.insert(0, '0')
    return vocabulary


def init_word2idx(vocabulary):
    return {val: key for key, val in enumerate(vocabulary)}


def init_idx2word(vocabulary):
    return {key: val for key, val in enumerate(vocabulary)}


def data_generator(sketch_id_code_set, word2idx, batch_size, maxlen, vocabulary=None):
    """data generator, intended to be used in a call to model.fit_generator()"""

    X_img = np.zeros((batch_size, *IMAGE_SIZE))
    X_seq = []
    Y_seq = []
    n = 0

    # loop for ever over images
    while True:
        keys = list(sketch_id_code_set.keys())
        np.random.shuffle(keys)
        data_set = [(key, sketch_id_code_set[key]) for key in keys]
        
        for sketch_id, code in data_set:
            # load sketch
            sketch = get_preprocessed_img(PATH_TO_DATA+sketch_id+'.png')
            X_img[n] = sketch

            # encode the sequence
            y_seq = [word2idx[word] for word in tokenize_dsl_code(code)] + [word2idx['<END>']]
            x_seq = [word2idx['<START>']] + y_seq[:-1]

            Y_seq.append(y_seq)
            X_seq.append(x_seq)

            n += 1
            # yield the batch data
            if n == batch_size:
                X_seq = pad_sequences(X_seq, maxlen=maxlen, padding='post')
                Y_seq = pad_sequences(Y_seq, maxlen=maxlen, padding='post')

                # One-hot
                Y_seq = [[
                    to_categorical(idx, len(vocabulary)) for idx in sent
                ] for sent in Y_seq]

                yield [[X_img, X_seq], np.array(Y_seq)]

                X_img = np.zeros((batch_size, *IMAGE_SIZE))
                X_seq = []
                Y_seq = []
                n = 0
