import os, sys

# allow importing from parent dir
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import pickle
import argparse
import numpy as np
from numpy.lib.function_base import average
from tqdm import tqdm
from utils.spanPooling import spanPooling
from utils.glove_utils import get_word_emb, get_phrase_emb, init_glove_data
from utils.utils import load_model
from sentence_transformers import SentenceTransformer
from sentence_transformers import models
from config.model_path import MODEL_PATH



def compute_emb_given_nested_data(text_list, model_path, spanRep=False):
    '''
        data: a list of sublist, each element a string, of the word to be distinguished
              In Turney, each sublist has 8 entries, 
              1 - bigram query, 2 - correct, 3 - component, 4 - component, 5/6/7/8 - other candidates
    '''
    if 'glove' in model_path:
        word2coeff_dict, average_emb = load_model(model_path)
    else:
        model = load_model(model_path)

    all_emb_list = []
    if 'glove' in model_path:
        for text_sublist in text_list:
            emb_list = []
            for entry in text_sublist:
                emb = get_phrase_emb(word2coeff_dict, entry, average_emb)
                emb_list.append(emb)
            all_emb_list.append(emb_list)
    else:
        for text_sublist in text_list:
            emb_list = model.encode( text_sublist, batch_size=len(text_sublist), show_progress_bar=False)
            all_emb_list.append(emb_list)
    return all_emb_list


def conduct_turney_test(data_list, all_emb_list):
    """
        data: a list of sublist, each element a string, of the word to be distinguished
              In Turney, each sublist has 8 entries, 
              1 - bigram query, 2 - correct, 3 - component, 4 - component, 5/6/7/8 - other candidates
        
        all_emb_list: a list of sublist, each element a np array, of the entry's emb
              In Turney, each sublist has 8 entries, 
              1 - bigram query, 2 - correct, 3 - component, 4 - component, 5/6/7/8 - other candidates
        
    """
    num_correct = 0    
    for idx, emb_list in enumerate(all_emb_list):
        text_list = data_list[idx]

        emb_array = np.concatenate( (emb_list[:2], emb_list[4:]), axis=0 )
        text_list = text_list[:2] + text_list[4:]
        query = emb_array[0, :]
        matrix = np.array( emb_array[1:,:])
        scores = np.dot(matrix, query)
        chosen = np.argmax(scores)

        if chosen == 0:
            num_correct += 1

    accuracy = num_correct / len(data_list)
    print(f'Accuracy on Turney = {accuracy}')

def main(args):
    # load the data for the turney task
    turney_data_fname = args.turney_fname
    with open(turney_data_fname, 'r') as f:
        content = f.readlines()
        data_list = []
        for line in content:
            components = line.strip('\n').split(' | ')
            data_list.append(components)

    for model_name, model_path in MODEL_PATH.items():
        print(model_name)
        all_emb_list = compute_emb_given_nested_data(data_list, model_path)
        conduct_turney_test(data_list, all_emb_list)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--turney_fname', type=str, 
            default='')
    args = parser.parse_args()
    main(args)