import os, sys

# allow importing from parent dir
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import pickle
import argparse
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sentence_transformers import models
from torch import mode, nn
from scipy.stats.stats import pearsonr
import torch
from utils.spanPooling import spanPooling
from utils.utils import load_model
from utils.glove_utils import get_phrase_emb
from config.model_path import MODEL_PATH

# The following function is modified from Yu and Ettinger, ACL 2020 
# (https://github.com/yulang/phrasal-composition-in-transformers/tree/master/src)
def main(args):
    data_fname = args.bird_fname

    # iterate through each model to be tested
    for model_name, model_path in MODEL_PATH.items():
        print(model_name)

        text_list = []
        scores = []

        # read in BiRD data
        bird_handler = open(data_fname, "r")
        for line_no, line in enumerate(bird_handler):
            if line_no == 0:
                # skip header
                continue
            words = line.rstrip().split("\t")
            p1, p2, score = words[1], words[2], float(words[-2])
            text_list.append( [p1, p2] )
            scores.append(score)

        all_emb_list = []

        # load the model and perform inference on data to obtain embeddings
        if 'glove' not in model_name:
            model = load_model(model_path)

            for text_sublist in text_list:
                emb_list = model.encode( text_sublist, batch_size=len(text_sublist), show_progress_bar=False)
                all_emb_list.append(emb_list)
        else:
            word2coef_dict, average_emb = load_model(model_path)
            for text_sublist in text_list:
                emb_list = []
                for term in text_sublist:
                    emb = get_phrase_emb(word2coef_dict, term, average_emb)
                    emb_list.append(emb)
                all_emb_list.append(emb_list)


        # Following Yu and Ettinger, which uses Cosine similarity on BiRD task evaluation
        cos_sim = nn.CosineSimilarity(dim=0)
        normalized = True
        cos_sim_list = []
        for emb_list in all_emb_list:
            [e1, e2] = emb_list
            sim = cos_sim(torch.tensor(e1), torch.tensor(e2))
            if normalized:
                sim = (sim + 1) / 2.0
            cos_sim_list.append(sim.item())
        cor, _ = pearsonr(cos_sim_list, scores)
        print(cor)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bird_fname', type=str, 
            default='')
    args = parser.parse_args()
    main(args)

    