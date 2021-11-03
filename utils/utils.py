import os, sys

# allow importing from parent dir
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers import models
from utils.spanPooling import spanPooling
from utils.glove_utils import get_word_emb, get_phrase_emb, init_glove_data
from config.model_path import MODEL_PATH, GLOVE_FILE_PATH


def load_model(model_path, spanRep=False):
    if 'glove' in model_path:
        glove_dict_fname = os.path.join( model_path, 'glove_dict.pkl')
        average_emb_fname = os.path.join( model_path, 'default_value.npy')
        if not (os.path.exists(glove_dict_fname) and os.path.exists(average_emb_fname)) : 
            # initialize the glove dictionary if never used
            init_glove_data(GLOVE_FILE_PATH, model_path)
        # glove model is loaded in the the form of a dictionary and a default value for oov
        with open( glove_dict_fname, 'rb') as f:
            word2coeff_dict = pickle.load(f)
        average_emb = np.load( average_emb_fname)
        return word2coeff_dict, average_emb

    if model_path == 'bert-base-uncased' or model_path == 'bert-large-uncased':
        word_embedding_model = models.Transformer(model_path)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    elif 'spanbert' in model_path:
        if spanRep:
            word_embedding_model = models.Transformer(model_path)
            pooling_model = spanPooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode_span=True)
            model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        else:
            word_embedding_model = models.Transformer(model_path)
            pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
            model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    else:
        model = SentenceTransformer(model_path)
    return model
