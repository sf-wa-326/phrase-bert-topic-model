'''

    This script contains helper methods for the topic model, eg: computing losses, forward pass in an epoch


'''

import torch
import numpy as np
import time, pickle
import random, os
from tqdm import tqdm


def l2_normalize_batch(x):
    return x.div( x.norm(p=2, dim=1, keepdim=True) )


# use this to compute objective function, where the target distribution has 1/X probability for each of the X words in the input
def soft_cross_entropy(input, target):
    return torch.mean(torch.sum(-target * torch.nn.functional.log_softmax(input, dim=1), dim=1))

def compute_triplet_loss(anchor, positive, negative, margin):
    b_size, dim_size = anchor.size()

    anchor_normalized = l2_normalize_batch(anchor)
    positive_normalized = l2_normalize_batch(positive)
    negative_normalized = l2_normalize_batch(negative)


    positive_dot = torch.bmm(anchor_normalized.view(b_size, 1, dim_size), positive_normalized.view(b_size, dim_size, 1))
    negative_dot = torch.bmm(anchor_normalized.view(b_size, 1, dim_size), negative_normalized.view(b_size, dim_size, 1))

    losses = torch.nn.functional.relu(1.0 + negative_dot - positive_dot)
    return losses.mean()

def run_epoch(net, optim, batch_intervals_train, uid_input_vector_list, uid_input_vector_list_neg, args, train, h_model, epoch, freeze_begin_epoch):
    device = args.device
    triplet_loss_margin = args.triplet_loss_margin
    triplet_loss_weight = args.triplet_loss_weight
    ortho_weight = args.ortho_weight
    neighbour_loss_weight = args.neighbour_loss_weight
    offset_loss_weight = args.offset_loss_weight

    ep_loss = 0.
    ep_tri_loss = 0.
    ep_re_loss = 0.
    ep_or_loss = 0.
    ep_world_class_loss = 0.
    ep_off_loss = 0.
    ep_nei_loss = 0.
    start_time = time.time()
    net.train()

    # After reaching freeze_begin_epoch, freeze all weights but X
    if epoch == freeze_begin_epoch:
        for name, param in net.named_parameters():
            if name == 'X': # allow gradients update for the sub topic offsets
                param.requires_grad = True
            else: # freeze gradient updates for all other topic offsets
                param.requires_grad = False

    if epoch < freeze_begin_epoch:
        ortho_weight = 0.
        neighbour_loss_weight = 0.
        offset_loss_weight = 0.


    for b_idx, (start, end) in enumerate(batch_intervals_train):
        # print(start, end)

        batch_data = uid_input_vector_list[start:end]
        batch_input_vec = [uid_vec_pair[1] for uid_vec_pair in batch_data]
        try:
            batch_data_t = torch.FloatTensor(np.array(batch_input_vec)).to(device)
        except Exception as e:
            print('error')

        batch_data_neg = uid_input_vector_list_neg[start:end]
        batch_data_neg_t = torch.FloatTensor(np.array(batch_data_neg)).to(device)

        recomb = net(batch_data_t, epoch, freeze_begin_epoch)


        triplet_loss = triplet_loss_weight * compute_triplet_loss(recomb, batch_data_t, batch_data_neg_t, triplet_loss_margin)

        # construct world classification target
        world_targets = [vec_id_pair[1] for vec_id_pair in batch_data]
        world_targets_t = torch.LongTensor(np.array(world_targets)).to(device)



        # compute world loss using the doc here: https://pytorch.org/docs/stable/nn.html#crossentropyloss

        # compute orthogonality penalty on dictionary
        X = net.X
        offset_loss = torch.tensor([0.0], device=args.device, requires_grad=True)
        neighbour_loss = torch.tensor([0.0], device=args.device, requires_grad=True)
        if len(X.shape)>2 and h_model == 1:
            # neighbour_loss
            Xp = X.permute((0, 2, 1))
            y = torch.bmm(X, Xp)
            e = torch.eye(y.shape[-1])
            new_e = torch.cat( args.num_topics * [e.unsqueeze(0)])
            z = (y - new_e.to(device)) ** 2
            neighbour_loss = - ortho_weight * 0.01 * torch.sum(z)
            X = X.view(-1, X.shape[-1])
        elif len(X.shape)>2 and h_model == 2:
            # neighbour_loss
            Xp = X.permute((0, 2, 1))
            y = torch.bmm(X, Xp)
            e = torch.eye(y.shape[-1])
            new_e = torch.cat( args.num_topics * [e.unsqueeze(0)])
            z = (y - new_e.to(device)) ** 2
            neighbour_loss = - neighbour_loss_weight * torch.sum(z)
            X = X.view(-1, X.shape[-1]) # prepare X for orthogonality loss
            # will add a loss to enforce small magnitude on X
            X = X.view(-1, X.shape[-1])
            Xr_squared = X ** 2
            X_l2 = torch.sum(Xr_squared)    
            offset_loss = offset_loss_weight * X_l2
        else:
            pass

        X = torch.nn.functional.normalize(X, dim=0)
        ortho_loss = ortho_weight * torch.sum((torch.mm(X, X.t()) - \
                                               torch.eye(X.size()[0]).to(device)) ** 2)



        batch_loss = triplet_loss + ortho_loss + neighbour_loss + offset_loss

        if train: # at training time we perform gradient updates
            batch_loss.backward()
            optim.step()
            optim.zero_grad()

        # else: # at validation time we compute prediction accuracies


        ep_loss += batch_loss.item()
        ep_tri_loss += triplet_loss.item()
        ep_or_loss += ortho_loss.item()
        ep_off_loss += offset_loss.item()
        ep_nei_loss += neighbour_loss.item()

    ep_loss = ep_loss / len(batch_intervals_train)
    ep_tri_loss = ep_tri_loss / len(batch_intervals_train)
    ep_or_loss = ep_or_loss / len(batch_intervals_train)
    ep_off_loss = ep_off_loss / len(batch_intervals_train)
    ep_nei_loss = ep_nei_loss / len(batch_intervals_train)
    signature = 'TRAIN' if train == True else 'VALID'

    ep_info = '[%s] loss: %0.4f, %0.4f, %0.4f, %0.4f, %0.4f (all, tri, or, off, nei), time: %0.2f s' % (signature,
        ep_loss, ep_tri_loss, ep_or_loss, ep_off_loss, ep_nei_loss, time.time() - start_time)

    print(ep_info)



def smart_rank_vocab_for_topics(model=None, word_embedding_matrix=None, to_be_removed=None, sorted_ids=None, top_activation_J=10, return_format='str'):
        # if self.mode == 'glove':
        #     id2word_dict = self.vrev
        #     vocab_input = [[i] for i in range(len(id2word_dict))]
        #     vocab_input_t = torch.LongTensor(vocab_input).to(self.device)
        # if self.mode == 'bert':

    if top_activation_J == 0: # this means that we do not use top J reranking and therefore get the top 10 candidates
        top_activation_J = 10
        sorted_ids = None
    
    if True:
        vocab_input = word_embedding_matrix
        vocab_input_t = torch.FloatTensor(vocab_input).to(model.device)
        topics_print_list = []
        with torch.no_grad():
            # in torch
            dict_queries = model.get_query(vocab_input_t)
            topic_dict = torch.nn.functional.normalize(model.X, dim=1)
            scores_over_vocab = torch.mm(topic_dict, dict_queries.t())  # K by num_vocab
            prob_over_vocab = torch.nn.functional.softmax(scores_over_vocab, dim=1)  # K by num_vocab
            prob_over_vocab_np = prob_over_vocab.cpu().detach().numpy()

            if len(to_be_removed) > 0:
                prob_over_vocab_np[:, to_be_removed] = 0.
                # print('removed some words')


            top_probable = np.argsort(prob_over_vocab_np)
            top_J = top_probable[:, -top_activation_J:]

            for topic_id in range(top_J.shape[0]):
                if sorted_ids is not None:
                    top_10_words_list = []
                    for id in sorted_ids:
                        if id in top_J[topic_id]:
                            top_10_words_list.append(model.vrev[id])
                        if len(top_10_words_list) == 10:
                            break
                else:
                    top_10_words_list = [model.vrev[x] for x in top_J[topic_id]]
                
                if return_format == 'str':
                    top_10_words_joined = ', '.join(top_10_words_list)
                    topic_print = f'topic {topic_id} : ' + top_10_words_joined
                    # print(topic_print)
                    topics_print_list.append(topic_print)
                elif return_format == 'list':
                    topics_print_list.append(top_10_words_list)
                else:
                    print('wrong format')

        return topics_print_list


def text_to_topic(input_vector_list, model, device, batch_size=400):
    batch_intervals = [(start, start + batch_size) for start in range(0, len(input_vector_list), batch_size)]

    result = np.empty((0), dtype=int)
    for b_idx, (start, end) in enumerate( tqdm(batch_intervals) ):
        batch_data = input_vector_list[start:end]
        batch_data_t = torch.FloatTensor(np.array(batch_data)).to(device)
        with torch.no_grad():
            scores = model.evaluate_topics(batch_data_t)
        _, ind = torch.max(scores, 1)
        ind_np = ind.data.cpu().numpy()
        result = np.concatenate((result, ind_np))
    return result



def rank_topics_by_percentage(topic_pred_list):
    from collections import Counter
    topics_counter = Counter(topic_pred_list)
    num_topic = len(topics_counter)

    topic_id_ranked = []
    topic_percentage_ranked = []    
    
    for rank, (topic_id, cnt) in enumerate( topics_counter.most_common() ):
        # print(f'Rank: {rank}, Topic: {topic_id}, Percentage { cnt/ len(topic_pred_list) }')
        topic_id_ranked.append(topic_id)
        perc = str( round( 100 * cnt/ len(topic_pred_list), 2) )
        topic_percentage_ranked.append( perc )
    
    return topic_id_ranked, topic_percentage_ranked