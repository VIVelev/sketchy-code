__all__ = [
    'get_sketch_id_code_pair',
    'init_sketch_id_code_map',
    'load_vocabulary',
    'init_word2idx',
    'init_idx2word',
]


def get_sketch_id_code_pair(path_to_code):
    sketch_id = path_to_code.split('/')[3].split('.')[0]
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
    with open('../vocabulary.txt', 'r') as f:
        vocabulary = f.read().split()

    vocabulary.insert(0, '0')
    return vocabulary

def init_word2idx(vocabulary):
    return {val: key for key, val in enumerate(vocabulary)}

def init_idx2word(vocabulary):
    return {key: val for key, val in enumerate(vocabulary)}
