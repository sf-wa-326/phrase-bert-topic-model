'''

Modified from version 0.3.3 of Sentence Transformers, from Reimers, Nils and Gurevych, Iryna, 2019
(https://github.com/UKPLab/sentence-transformers/tree/v0.3.3)
We finetune the phrase embedding using paraphrases and phrase-context match


'''

from sentence_transformers import SentenceTransformer, SentencesDataset, LoggingHandler, losses, models
from torch.utils.data import DataLoader
from sentence_transformers.readers import TripletReader
from sentence_transformers.evaluation import TripletEvaluator
from datetime import datetime

import torch
import csv, os
import logging
import argparse
import json
parser = argparse.ArgumentParser()
parser.add_argument('--input_data_path', type=str, 
    default='')
parser.add_argument('--train_data_file', type=str, 
    default='')
parser.add_argument('--valid_data_file', type=str, 
    default='')
parser.add_argument('--input_model_path', type=str, 
    default='bert-base-nli-stsb-mean-tokens')
parser.add_argument('--output_model_path', type=str, 
    default='')
args = parser.parse_args()

print(torch.cuda.is_available())



logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])


#You can specify any huggingface/transformers pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base
# we use model_name = 'bert-base-nli-stsb-mean-tokens'
model_name = args.input_model_path
num_epochs = 1


### Create a torch.DataLoader that passes training batch instances to our model
train_batch_size = 16
triplet_reader = TripletReader( args.input_data_path,
                                s1_col_idx=0, 
                                s2_col_idx=1, 
                                s3_col_idx=2, 
                                delimiter='\t', 
                                quoting=csv.QUOTE_MINIMAL, 
                                has_header=True)



# model_name = 'bert-base-nli-stsb-mean-tokens'
model = SentenceTransformer(model_name)


logging.info("Read Triplet train dataset")
train_dataset = SentencesDataset(examples=triplet_reader.get_examples(args.train_data_file, max_examples=0), model=model)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
train_loss = losses.TripletLoss(model=model)

logging.info("Read Triplet dev dataset")
evaluator = TripletEvaluator.from_input_examples(triplet_reader.get_examples(args.valid_data_file, max_examples=0), name='dev')


warmup_steps = int(len(train_dataset) * num_epochs / train_batch_size * 0.1) #10% of train data


# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=1000,
          warmup_steps=warmup_steps,
          output_path=args.output_model_path)

print('done')