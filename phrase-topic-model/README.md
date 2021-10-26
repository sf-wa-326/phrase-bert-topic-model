# Phrase-BERT-based Neural Topic Model (PNTM)

As a case study, we show that Phrase-BERT embeddings can be easily integrated with a simple autoencoder to build a phrase-based neural topic model. We follow the architecture of the neural topic model (a unigram-based topic model) used in [Iyyer et al 2016](https://aclanthology.org/N16-1180/), and [Akoury et al 2020](https://arxiv.org/abs/2010.01717). We simply replace the embedding model (glove) in the previous works by Phrase-BERT, which gives topic models that can describe topics using a mixture of unigrams and phrases.

To run PNTM on a provided example of wikipedia dataset (as described in our [paper](https://arxiv.org/abs/2109.06304)), please following the following steps:

1. **Step 1**: Download the wikipedia text [dataset](https://storage.googleapis.com/phrase-bert/topic-model/text_list.json) and also the vocabularies, which are the [word2id_dictionary](https://storage.googleapis.com/phrase-bert/topic-model/combined_word2id_dict.pkl), [id2word_dictionary](https://storage.googleapis.com/phrase-bert/topic-model/combined_id2word_dict.pkl), [id2frequency_dictionary](https://storage.googleapis.com/phrase-bert/topic-model/id2freq_dict.pkl)).
````
>>> cd <topic-model-data-path>
>>> wget https://storage.googleapis.com/phrase-bert/topic-model/text_list.json
>>> wget https://storage.googleapis.com/phrase-bert/topic-model/combined_word2id_dict.pkl
>>> wget https://storage.googleapis.com/phrase-bert/topic-model/combined_id2word_dict.pkl
>>> https://storage.googleapis.com/phrase-bert/topic-model/id2freq_dict.pkl
````


2. **Step 2**: run the preprocessing script `preprocess.py`
````
>>> python -u preprocess.py \
    --topic_model_data_path <topic-model-data-path> \
    --emb_model_path <directory-to-phrasebert-model>
````
which produces the embedding matrix of vocabularies (unigrams and phrases), and the vector representation of text given.

3. **Step 3**: train the topic model using `run_topic_model.py`:
````
>>> python -u run_topic_model.py \
    --num_topics 50 \
    --num_epochs 300 \
    --random_seed 42 \
    --topic_model_data_path <topic-model-data-path> \
    --emb_model phrase-bert
````