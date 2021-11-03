import numpy as np
import os, pickle
from tqdm import tqdm
def get_word_emb(word2coef_dict, word, default_value):
    return word2coef_dict.get(word, default_value)

def get_phrase_emb(word2coef_dict, phrase, default_value):
    words = phrase.split(' ')
    embs = [ get_word_emb(word2coef_dict, word, default_value) for word in words ]
    return np.mean(embs, axis=0)

def init_glove_data(glove_fname, glove_outname):
    print(f'Constructing glove dictionary from {glove_fname} ...... ')
    word2coef_dict = {}
    running_sum = np.zeros((300,))
    with open(os.path.join(glove_fname), 'r') as f:
        for idx, line in enumerate(tqdm( list(f) )):
            values = line.split()
            word = ''.join(values[0:-300])
            coefs = np.asarray(values[-300:], dtype='float32')
            running_sum += coefs
            word2coef_dict[word] = coefs
        average_emb = running_sum / (idx + 1)

    with open( os.path.join(glove_outname, 'glove_dict.pkl'), 'wb') as f:
        pickle.dump( word2coef_dict, f)
    np.save( os.path.join(glove_outname, 'default_value'), average_emb)
    print(f'Glove dictionary saved at {glove_outname}')