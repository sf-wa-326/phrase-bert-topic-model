# Phrase-BERT: Improved Phrase Embeddings from BERT with an Application to Corpus Exploration

This is the official repository for the EMNLP 2021 long paper [Phrase-BERT: Improved Phrase Embeddings from BERT with an Application to Corpus Exploration](https://arxiv.org/abs/2109.06304). We provide code for training and evaluating Phrase-BERT in addition to the datasets used in the paper.

Update: the model is also available now on [Huggingface](https://huggingface.co/whaleloops/phrase-bert) thanks to the help from [whaleloops](https://github.com/whaleloops) and [nreimers](https://github.com/nreimers)!


## Setup
This repository depends on sentence-BERT version 0.3.3, which you can install from the source using:
````
>>> git clone https://github.com/UKPLab/sentence-transformers.git --branch v0.3.3
>>> cd sentence-transformers/
>>> pip install -e .

````

Also you can install sentence-BERT with `pip`:
````
>>> pip install sentence-transformers==0.3.3
````

## Quick Start
The following example shows how to use a trained Phrase-BERT model to embed phrases into dense vectors.

First download and unzip our model.
````
>>> cd <local-directory-to-store-models>
>>> wget https://storage.googleapis.com/phrase-bert/phrase-bert/phrase-bert-model.zip
>>> unzip phrase-bert-model.zip -d phrase-bert-model/
>>> rm phrase-bert-model.zip
````


Then load the Phrase-BERT model through the sentence-BERT interface:
````
from sentence_transformers import SentenceTransformer
model_path = '<path-to-phrase-bert-model>'
model = SentenceTransformer(model_path)
````

You can compute phrase embeddings using Phrase-BERT as follows:
````
phrase_list = [ 'play an active role', 'participate actively', 'active lifestyle']
phrase_embs = model.encode( phrase_list )
[p1, p2, p3] = phrase_embs
````

As in sentence-BERT, the default output is a list of numpy arrays:
````
for phrase, embedding in zip(phrase_list, phrase_embs):
    print("Phrase:", phrase)
    print("Embedding:", embedding)
    print("")
````

An example of computing the dot product of phrase embeddings:
````
import numpy as np
print(f'The dot product between phrase 1 and 2 is: {np.dot(p1, p2)}')
print(f'The dot product between phrase 1 and 3 is: {np.dot(p1, p3)}')
print(f'The dot product between phrase 2 and 3 is: {np.dot(p2, p3)}')
````

An example of computing cosine similarity of phrase embeddings:
````
import torch 
from torch import nn
cos_sim = nn.CosineSimilarity(dim=0)
print(f'The cosine similarity between phrase 1 and 2 is: {cos_sim( torch.tensor(p1), torch.tensor(p2))}')
print(f'The cosine similarity between phrase 1 and 3 is: {cos_sim( torch.tensor(p1), torch.tensor(p3))}')
print(f'The cosine similarity between phrase 2 and 3 is: {cos_sim( torch.tensor(p2), torch.tensor(p3))}')
````

The output should look like:
````
The dot product between phrase 1 and 2 is: 218.43600463867188
The dot product between phrase 1 and 3 is: 165.48483276367188
The dot product between phrase 2 and 3 is: 160.51708984375
The cosine similarity between phrase 1 and 2 is: 0.8142536282539368
The cosine similarity between phrase 1 and 3 is: 0.6130303144454956
The cosine similarity between phrase 2 and 3 is: 0.584893524646759
````


## Evaluation
Given the lack of a unified phrase embedding evaluation benchmark, we collect the following five phrase semantics evaluation tasks, which are described further in our paper:

* Turney [[Download](https://storage.googleapis.com/phrase-bert/turney/data.txt) ]
* BiRD [[Download](https://storage.googleapis.com/phrase-bert/bird/data.txt)]
* PPDB [[Download](https://storage.googleapis.com/phrase-bert/ppdb/examples.json)]
* PPDB-filtered [[Download](https://storage.googleapis.com/phrase-bert/ppdb_exact/examples.json)]
* PAWS-short [[Download Train-split](https://storage.googleapis.com/phrase-bert/paws_short/train_examples.json) ] [[Download Dev-split](https://storage.googleapis.com/phrase-bert/paws_short/dev_examples.json) ] [[Download Test-split](https://storage.googleapis.com/phrase-bert/paws_short/test_examples.json) ]


Change `config/model_path.py` with the model path according to your directories and 
* For evaluation on Turney, run `python eval_turney.py`
* For evaluation on BiRD, run `python eval_bird.py`
* for evaluation on PPDB / PPDB-filtered / PAWS-short, run `eval_ppdb_paws.py` with:

    ````
    nohup python  -u eval_ppdb_paws.py \
        --full_run_mode \
        --task <task-name> \
        --data_dir <input-data-dir> \
        --result_dir <result-storage-dr> \
        >./output.txt 2>&1 &
    ````

## Train your own Phrase-BERT
If you would like to go beyond using the pre-trained Phrase-BERT model, you may train your own Phrase-BERT using data from the domain you are interested in. Please refer to 
`phrase-bert/phrase_bert_finetune.py`

The datasets we used to fine-tune Phrase-BERT are here: [training data csv file](https://storage.googleapis.com/phrase-bert/phrase-bert-ft-data/pooled_context_para_triples_p%3D0.8_train.csv) and [validation data csv file](https://storage.googleapis.com/phrase-bert/phrase-bert-ft-data/pooled_context_para_triples_p%3D0.8_valid.csv).

To re-produce the trained Phrase-BERT, please run:

    export INPUT_DATA_PATH=<directory-of-phrasebert-finetuning-data>
    export TRAIN_DATA_FILE=<training-data-filename.csv>
    export VALID_DATA_FILE=<validation-data-filename.csv>
    export INPUT_MODEL_PATH=bert-base-nli-stsb-mean-tokens 
    export OUTPUT_MODEL_PATH=<directory-of-saved-model>


    python -u phrase_bert_finetune.py \
        --input_data_path $INPUT_DATA_PATH \
        --train_data_file $TRAIN_DATA_FILE \
        --valid_data_file $VALID_DATA_FILE \
        --input_model_path $INPUT_MODEL_PATH \
        --output_model_path $OUTPUT_MODEL_PATH

## Citation:
Please cite us if you find this useful:
````
@inproceedings{phrasebertwang2021,
    author={Shufan Wang and Laure Thompson and Mohit Iyyer},
    Booktitle = {Empirical Methods in Natural Language Processing},
    Year = "2021",
    Title={Phrase-BERT: Improved Phrase Embeddings from BERT with an Application to Corpus Exploration}
}
````

