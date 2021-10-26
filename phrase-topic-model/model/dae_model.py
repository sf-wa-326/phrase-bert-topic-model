import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Function

from .base_model import BaseModel


class DictionaryAutoencoder(BaseModel):
    def __init__(self, net_params):
        super(DictionaryAutoencoder, self).__init__()

        # print("We got here")
        # store for interpretation
        self.vrev = net_params["vrev"]  # idx to word mapping
        self.mode = net_params["mode"]  # bert or GloVe

        # hyperparams
        self.vocab_size = net_params["embedding"].shape[0]

        self.d_emb = net_params["embedding"].shape[1]
        self.d_hid = net_params["d_hid"]
        self.K = net_params["num_rows"]  # number of topics
        self.device = net_params["device"]
        self.pred_world = net_params["pred_world"]
        if self.pred_world:
            self.num_world = net_params["num_world"]
            self.W_world = nn.Linear(self.d_emb, self.num_world)

        # glove params params
        self.embeddings = nn.Embedding(self.vocab_size, self.d_emb)
        self.embeddings.weight.data.copy_(torch.from_numpy(net_params["embedding"]))
        self.embeddings.weight.requires_grad = False

        # put an MLP on top of embeddings for more params
        self.W_proj = nn.Linear(self.d_emb, self.d_hid)  # bottleneck
        self.act = nn.ReLU()
        # self.dropout = nn.Dropout(p=0.2)
        self.word_dropout_prob = net_params["word_dropout_prob"]
        self.dropout = nn.Dropout(self.word_dropout_prob)

        # fully-connected layer to project back to emb. dimension
        self.W_att = nn.Linear(self.d_hid, self.d_emb)  # back to d_emb for attention


        # reconstruction output
        self.W_out = nn.Linear(self.d_emb, self.vocab_size)  # output matrix
        self.W_out.weight.data.copy_(torch.from_numpy(net_params["embedding"]))
        self.W_out.weight.requires_grad = False

        # predict world info
        if self.pred_world:
            self.W_world = nn.Linear(self.d_emb, self.num_world)

        # dictionary, each row is a "topic" embedding
        self.X = nn.Parameter(torch.randn(net_params["num_rows"], self.d_emb))

        # rows should initially be (semi) orthogonal
        torch.nn.init.orthogonal_(self.X)

    # print nearest neighbors of each row w/ cosine distance
    def interpret_dictionary(self):
        # get current embeddings and normalize them
        We = self.embeddings.weight.data.detach()
        We = torch.nn.functional.normalize(We, dim=1).cpu().numpy()
        topic_dict = torch.nn.functional.normalize(self.X, dim=1)

        X = topic_dict.detach().cpu().numpy()
        for i in range(self.K):
            desc = X[i]
            sims = We.dot(desc.T)
            ordered_words = np.argsort(sims)[::-1]
            desc_list = [self.vrev[w] for w in ordered_words[:10]]
            print("topic %d: %s" % (i, ", ".join(desc_list)))

    def get_query(self, batch):
        bsz, d_emb = batch.size()
        embs = self.dropout(batch)
        proj = self.dropout(self.act(self.W_proj(embs)))
        latent = proj

        # project back to embedding size
        dict_query = self.dropout(self.act(self.W_att(latent)))
        return dict_query

    def forward(self, batch, epoch, freeze_begin_epoch):
        dict_query = self.get_query(batch)
        # normalize dictionary
        topic_dict = torch.nn.functional.normalize(self.X, dim=1)

        # now compute attention over dictionary matrix X
        scores = torch.mm(dict_query, topic_dict.t())
        scores = torch.nn.functional.softmax(scores, dim=1)  # bsz x K

        # now get weighted aves
        recomb = torch.mm(scores, topic_dict)  # bsz x d_emb

        # use the weighted aves to classify world label
        if self.pred_world:
            world_logits = self.W_world(recomb)
            return recomb, world_logits

        return recomb

        # project to vocab size to predict original words in the input
        # recon_logits = self.W_out(recomb)  # bsz x len_voc

    def evaluate_topics(self, batch):
        # each example in a batch is of the form w1 w2 ... wn

        """
        :param batch: input in the form of word IDs (tensor)
        :return: the score, probability distribution, over all K topics
        """
        with torch.no_grad():
            bsz, d_emb = batch.size()

            # project back to embedding size
            dict_query = self.get_query(batch)

            # normalize dictionary
            topic_dict = torch.nn.functional.normalize(self.X, dim=1)

            # now compute attention over dictionary matrix X
            scores = torch.mm(dict_query, topic_dict.t())
            return torch.nn.functional.softmax(scores, dim=1)  # bsz x K

    def rank_vocab_for_topics(self, word_embedding_matrix, to_be_removed):

        vocab_input = word_embedding_matrix
        vocab_input_t = torch.FloatTensor(vocab_input).to(self.device)
        topics_print_list = []
        with torch.no_grad():
            # in torch
            dict_queries = self.get_query(vocab_input_t)
            topic_dict = torch.nn.functional.normalize(self.X, dim=1)
            scores_over_vocab = torch.mm(topic_dict, dict_queries.t())  # K by num_vocab
            prob_over_vocab = torch.nn.functional.softmax(
                scores_over_vocab, dim=1
            )  # K by num_vocab
            prob_over_vocab_np = prob_over_vocab.cpu().detach().numpy()

            if len(to_be_removed) > 0:
                prob_over_vocab_np[:, to_be_removed] = 0.0
                # print('removed some words')

            top_probable = np.argsort(prob_over_vocab_np)
            top_10 = top_probable[:, -10:]
            for topic_id in range(top_10.shape[0]):
                top_10_words_list = [self.vrev[x] for x in top_10[topic_id]]
                top_10_words_joined = ", ".join(top_10_words_list)
                topic_print = f"topic {topic_id} : " + top_10_words_joined
                print(topic_print)
                topics_print_list.append(topic_print)
        return topics_print_list

