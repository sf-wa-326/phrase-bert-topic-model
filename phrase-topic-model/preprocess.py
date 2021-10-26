'''

    Use sentence bert to construct word, phrase and text embedding

'''
import sys 
import pickle, json
import os
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import BertConfig, BertModel, BertTokenizer

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--topic_model_data_path", type=str, 
    default='')
parser.add_argument("--emb_model_path", type=str,
    default='')
args = parser.parse_args()


model = SentenceTransformer(args.emb_model_path)

# process the word embedding
with open( os.path.join(args.topic_model_data_path, 'combined_word2id_dict.pkl'), 'rb' ) as f:
    word2id_dict = pickle.load(f)
with open( os.path.join(args.topic_model_data_path, 'combined_id2word_dict.pkl'), 'rb' ) as f:
    id2word_dict = pickle.load(f)
word_list = [word for id, word in id2word_dict.items()]
print(f'loaded {len(word_list)} vocabs')
print(len(word_list))
result_list = model.encode(word_list, batch_size=8, show_progress_bar=True)
embedding_result = np.asarray(result_list)
np.save( os.path.join(args.topic_model_data_path, 'embedding_matrix_np'), embedding_result)

# process the text embedding
with open( os.path.join(args.topic_model_data_path, 'text_list.json'), 'r') as f:
    text_list = json.load(f)
print(len(text_list))
result_list = model.encode(text_list, batch_size=8, show_progress_bar=True)
with open(os.path.join(args.topic_model_data_path, 'text_rep.pkl'), 'wb') as f:
    pickle.dump(result_list, f)

print('Done')

