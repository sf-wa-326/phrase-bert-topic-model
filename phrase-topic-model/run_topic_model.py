import pickle
import os
import numpy as np

# import dae_model
from model.dae_model import DictionaryAutoencoder
import random
import argparse
from model_utils import run_epoch, text_to_topic, rank_topics_by_percentage
import torch

parser = argparse.ArgumentParser()

# parameters to tune [can be changed]
parser.add_argument("--batch_size", type=int, default=100)
parser.add_argument("--num_epochs", type=int, default=2)
parser.add_argument("--num_topics", type=int, default=50)
parser.add_argument("--num_sub_topics", type=int, default=0)
parser.add_argument("--h_model", type=int, default=2)
parser.add_argument("--dropout_rate", type=float, default=0.2)
parser.add_argument("--d_hidden", type=int, default=100)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--use_norm_we", help="normalizing the glove embedding", action="store_true")
parser.add_argument("--emb_model", type=str, default="phrase-bert")
parser.add_argument("--freq_threshold", type=int, default=5)

parser.add_argument("--triplet_loss_margin", type=float, default=1.0)
parser.add_argument("--ortho_weight", type=float, default=1e-5)
parser.add_argument("--neighbour_loss_weight", type=float, default=1e-7)
parser.add_argument("--offset_loss_weight", type=float, default=1e-4)
parser.add_argument("--triplet_loss_weight", type=float, default=1.0)
parser.add_argument("--freeze_begin_epoch", type=int, default=200)

parser.add_argument("--device_id", type=int, default=0)
parser.add_argument("--num_negative_samples", type=int, default=5)
parser.add_argument("--random_seed", type=int, default=42)
parser.add_argument("--topic_model_data_path", type=str, default="")
args = parser.parse_args()


random.seed(args.random_seed)
np.random.seed(args.random_seed)
torch.manual_seed(args.random_seed)

device = torch.device("cuda:" + str(args.device_id))
args.device = device


print(args)
with open(os.path.join(args.topic_model_data_path, "combined_word2id_dict.pkl"), "rb") as f:
    word2id_dict = pickle.load(f)
with open(os.path.join(args.topic_model_data_path, "combined_id2word_dict.pkl"), "rb") as f:
    id2word_dict = pickle.load(f)
embedding_matrix_np = np.load(
    os.path.join(args.topic_model_data_path, f"embedding_matrix_np.npy")
)
print(f"Loaded word embedding from {args.topic_model_data_path}")
print(f"Loaded vocab size of {len(word2id_dict)} (including phrases)")


# word frequency and filter info

# compute the length (in n-grams)
len_words = [0] * len(id2word_dict)
for (id, word) in id2word_dict.items():
    len_words[id] = len(word.split(' '))
indices_to_remove_based_on_len = [
            id
            for id, word_len in enumerate(len_words)
            if (word_len > 6 )
        ]

if os.path.exists(os.path.join(args.topic_model_data_path, "freq_result.pkl")):
    with open(os.path.join( args.topic_model_data_path, 'id2freq_dict.pkl' ), 'rb' ) as f:
        id2freq_dict = pickle.load(f)
    sorted_ids = [k for k, v in sorted(id2freq_dict.items(), key=lambda item: item[1])]
    sorted_ids.reverse()
    indices_to_remove_based_on_freq = [k for k, v in id2freq_dict.items() if v <= args.freq_threshold ]

    to_be_removed = list(set( indices_to_remove_based_on_freq + indices_to_remove_based_on_len ))
else:
    to_be_removed = indices_to_remove_based_on_len



print(f"Building sentence model by using {args.emb_model} as embedding model")

with open( os.path.join( args.topic_model_data_path, f"text_rep.pkl" ), "rb" ) as f:
    text_rep_list = pickle.load(f)
    uid_input_vector_list = [
            (i, text_rep_list[i]) for i in range(len(text_rep_list))
        ]
    print(f"Computed {len(uid_input_vector_list)} positive examples")

uid_input_vector_list_neg = []
indices = list(range(len(uid_input_vector_list)))
num_neg_samples = args.num_negative_samples
for idx in range(len(uid_input_vector_list)):
    indices_candidate = indices
    neg_indices = random.sample(indices_candidate, num_neg_samples)
    neg_samples = [uid_input_vector_list[neg_i][1] for neg_i in neg_indices]
    neg_vector = np.mean(neg_samples, axis=0)
    uid_input_vector_list_neg.append(neg_vector)
print(f"Computed {len(uid_input_vector_list_neg)} negative examples")

# training loop
# normalize glove embeddings
if args.use_norm_we:
    pass

# set up hyperparameters
net_params = {}
net_params["mode"] = "bert"
net_params["embedding"] = embedding_matrix_np
net_params["d_hid"] = args.d_hidden
net_params["num_rows"] = args.num_topics  # number of topics
net_params["num_sub_topics"] = args.num_sub_topics
net_params["word_dropout_prob"] = args.dropout_rate
net_params["vrev"] = id2word_dict  # idx to word map
net_params["device"] = device
net_params["pred_world"] = False


net = DictionaryAutoencoder(net_params=net_params)
net.to(device)

    # training specs
num_epochs = args.num_epochs
batch_size = args.batch_size
ortho_weight = args.ortho_weight
world_clas_weight = 0.0
optim = torch.optim.Adam(net.parameters(), lr=args.lr)
interpret_interval = int(np.ceil(num_epochs / 10))
h_model = args.h_model

# iterating through batches
batch_intervals = [
    (start, start + batch_size)
    for start in range(0, len(uid_input_vector_list), batch_size)
]
    # batch_intervals = batch_intervals[:100]
split = int(np.ceil(len(batch_intervals) * 0.9))
batch_intervals_train = batch_intervals[:split]
batch_intervals_valid = batch_intervals[split:]

print("\n" + "=" * 70)
for epoch in range(num_epochs):

    # training
    net.train()
    train_mode = True
    print(f"Epoch {epoch}")
    run_epoch(
            net,
            optim,
            batch_intervals_train,
            uid_input_vector_list,
            uid_input_vector_list_neg,
            args,
            train_mode,
            h_model,
            epoch,
            args.freeze_begin_epoch
    )

    # validation
    net.eval()
    train_mode = False
    with torch.no_grad():
        run_epoch(
                net,
                optim,
                batch_intervals_valid,
                uid_input_vector_list,
                uid_input_vector_list_neg,
                args,
                train_mode,
                h_model,
                epoch,
                args.freeze_begin_epoch
        )

    if (epoch) % interpret_interval == 0:
        print("Topics with probability argmax")
        topics_print_list = net.rank_vocab_for_topics(
                    word_embedding_matrix=embedding_matrix_np,
                    to_be_removed=to_be_removed,
                )
        print("=" * 70)
    print()
    print()
    print()
    print("=" * 70)

print("Finally after training")
net.eval()

print("Topics with probability argmax")
topics_print_list = net.rank_vocab_for_topics(
            word_embedding_matrix=embedding_matrix_np, to_be_removed=to_be_removed
    )
print("=" * 70)

# after training we evaluate all the topics percentage in the dataset and rank the topics by percentage
uid_list, vector_list = zip(*uid_input_vector_list)
topic_pred_list = text_to_topic(vector_list, net, device)

topic_id_ranked, topic_percentage_ranked = rank_topics_by_percentage( topic_pred_list )

for rank, (topic_id, topic_percentage) in enumerate( zip(topic_id_ranked, topic_percentage_ranked)):
    print(
            f"Rank: {rank}, Topic_id: {topic_id}, Topic Words: {topics_print_list[topic_id]}, \
            Topic Percentage: {topic_percentage}"
        )


with open( os.path.join( args.topic_model_data_path, 'topic_model.pt'), "wb") as f:
    torch.save(net, f)
    print(f"Saved model at { os.path.join(args.topic_model_data_path) }")


