from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time
import misc.utils as utils
from collections import OrderedDict
from functools import partial

import math
import torch
import torch.nn.functional as F
from torch import multiprocessing as mp
from multiprocessing.managers import BaseManager

import sys
sys.path.append("cider")
from pyciderevalcap.ciderD.ciderD import CiderD
sys.path.append("coco-caption")
from pycocoevalcap.bleu.bleu import Bleu

CiderD_scorer = None
Bleu_scorer = None
#CiderD_scorer = CiderD(df='corpus')

def init_scorer(cached_tokens):
    global CiderD_scorer
    CiderD_scorer = CiderD_scorer or CiderD(df=cached_tokens)
    global Bleu_scorer
    Bleu_scorer = Bleu_scorer or Bleu(4)

def array_to_str(arr):
    out = ''
    for i in range(len(arr)):
        out += str(arr[i]) + ' '
        if arr[i] == 0:
            break
    return out.strip()

def get_self_critical_reward(model, fc_feats, att_feats, att_masks, data_gts, gen_result, opt):
    batch_size = gen_result.size(0)# batch_size = sample_size * seq_per_img
    seq_per_img = batch_size // len(data_gts)
     
    # get greedy decoding baseline
    model.eval()
    with torch.no_grad():
        greedy_res, _ = model(fc_feats, att_feats, att_masks=att_masks, mode='sample')
    model.train()

    res = OrderedDict()
    
    gen_result = gen_result.data.cpu().numpy()
    greedy_res = greedy_res.data.cpu().numpy()
    for i in range(batch_size):
        res[i] = [array_to_str(gen_result[i])]
    for i in range(batch_size):
        res[batch_size + i] = [array_to_str(greedy_res[i])]

    gts = OrderedDict()
    for i in range(len(data_gts)):
        gts[i] = [array_to_str(data_gts[i][j]) for j in range(len(data_gts[i]))]

    res_ = [{'image_id':i, 'caption': res[i]} for i in range(2 * batch_size)]
    res__ = {i: res[i] for i in range(2 * batch_size)}
    gts = {i: gts[i % batch_size // seq_per_img] for i in range(2 * batch_size)}
    if opt.cider_reward_weight > 0:
        _, cider_scores = CiderD_scorer.compute_score(gts, res_)
        print('Cider scores:', _)
    else:
        cider_scores = 0
    if opt.bleu_reward_weight > 0:
        _, bleu_scores = Bleu_scorer.compute_score(gts, res__)
        bleu_scores = np.array(bleu_scores[3])
        print('Bleu scores:', _[3])
    else:
        bleu_scores = 0
    scores = opt.cider_reward_weight * cider_scores + opt.bleu_reward_weight * bleu_scores

    scores = scores[:batch_size] - scores[batch_size:]

    rewards = np.repeat(scores[:, np.newaxis], gen_result.shape[1], 1)

    return rewards

def get_arsk_loss_cuda(model,  fc_feats, att_feats, labels, att_masks, data_gts):

    # run the true trajectory
    model.train()
    seq, seqlogits, eff_length, states = model(fc_feats, att_feats, att_masks, mode='arm_sample')

    seq_length, batch_size, vocab_size = seqlogits.size()

    pi = torch.from_numpy(np.random.dirichlet(np.ones(vocab_size), batch_size*seq_length))\
        .reshape(seq_length*batch_size, vocab_size).float()

    # get the ground truth for computing cider score
    gts = OrderedDict()
    for i in range(len(data_gts)):
        gts[i] = [array_to_str(data_gts[i][j]) for j in range(len(data_gts[i]))]

    seq_per_img = batch_size // len(data_gts)

    num_states = min(seq_length + 1, len(states))

    num_ref = 50
    ref_cat = np.random.choice(range(vocab_size), num_ref, replace=False)

    with torch.no_grad():

        # Computing the pseudo_action
        phi_c = seqlogits.reshape(seq_length*batch_size, vocab_size)
        pi_c = pi.cuda()

        R_cat = torch.from_numpy(ref_cat).long().cuda()

        t0 = time.time()
        pseudo_action = pseudo_action_multiple_j(phi_c, R_cat, pi_c).reshape(num_ref, seq_length, batch_size, vocab_size)
        t1 = time.time()
        print('time for computing pseudo_action: ', str(t1-t0))

        pseudo_action = pseudo_action.cpu()
        seq = seq.cpu()
        pi = pi.reshape(seq_length, batch_size, vocab_size)

        torch.cuda.empty_cache()

        gars = torch.zeros((num_ref, seq_length, batch_size, vocab_size)).float()

        timer = [0,0,0]
        sample_times = 0
        for t in range(1, num_states):
            t2 = time.time()
            p_a_t = pseudo_action[:, t - 1, :, :]

            finished_b_id = torch.tensor([], dtype=torch.uint8)
            u_p_a_t = torch.tensor([]).long()
            b_id_index = torch.tensor([]).long()
            b_id_length = []
            valid_b = []

            for b in range(batch_size):

                u_p_a_t_b = torch.unique(p_a_t[:, b, :].cuda()).cpu()
                u_p_a_t_b_length = u_p_a_t_b.size(0)
                if u_p_a_t_b_length == 1:
                    continue
                valid_b.append(b)
                u_p_a_t = torch.cat((u_p_a_t, u_p_a_t_b))
                b_id_index = torch.cat((b_id_index, torch.ones(u_p_a_t_b_length).long() * b))
                b_id_length.append(u_p_a_t_b_length)
                #inverse_index.append(u_p_a_t_b_inverse_id)

                if int(seq[b, t - 2]) == 0:
                    finished_b_id = torch.cat((finished_b_id, finished_b_id.new_ones(u_p_a_t_b_length)))
                else:
                    finished_b_id = torch.cat((finished_b_id, finished_b_id.new_zeros(u_p_a_t_b_length)))

            u_p_a_t_length = u_p_a_t.size(0)
            if u_p_a_t_length == 0:
                continue
            t2_2 = time.time()
            # obtain the its, states and seqs according to unique pseudo_actions
            state_h, state_c = states[t]
            state_h_arm = state_h[:, b_id_index, :]
            state_c_arm = state_c[:, b_id_index, :]
            #torch.cuda.empty_cache()
            p_state_t = (state_h_arm.cuda(), state_c_arm.cuda())

            p_seq_t = seq.new_zeros((u_p_a_t_length, seq_length))
            if t == 1:
                p_seq_t[:, 0] = u_p_a_t
            else:
                p_seq_t[:, :t - 1] = seq[b_id_index, :t - 1]
                p_seq_t[:, t - 1] = u_p_a_t

            p_it_t = u_p_a_t.cuda()
            
            t2_5 = time.time()
            timer[0] += t2_2 - t2

            sampled_seq_t = model(t, p_it_t, p_state_t, p_seq_t, mode='arm_pseudo_sample')
            #torch.cuda.empty_cache()
            t3 = time.time()
            timer[1] += t3 - t2_5
            sample_times += b_id_index.size(0)
            if finished_b_id.sum() > 0:
                sampled_seq_t[finished_b_id] = seq[b_id_index[finished_b_id]]

            gts_t = {i: gts[int(b_id_index[i]) // seq_per_img] for i in range(b_id_index.size(0))}

            sampled_seq_t = sampled_seq_t.data.numpy()
            res_t = [{'image_id': i, 'caption': [array_to_str(sampled_seq_t[i])]} for i in range(len(sampled_seq_t))]
            t3_2 = time.time()

            _, rewards_t = CiderD_scorer.compute_score(gts_t, res_t)
            t3_5 = time.time()
            rewards_t = torch.from_numpy(rewards_t).float().split(b_id_length)
            unique_actions = u_p_a_t.split(b_id_length)    

            for i in range(len(valid_b)):
                reward = torch.zeros(vocab_size)
                reward[unique_actions[i]] = rewards_t[i]
                b = valid_b[i]
                reward_t_b = reward[p_a_t[:, b, :]]
                gars[:, t-1, b, :] = (reward_t_b - reward_t_b.mean(dim=1).unsqueeze(1)) * (1 - pi[t - 1, b, ref_cat] * vocab_size).unsqueeze(1)

            t4 = time.time()
            timer[2] += t3_5 - t3_2
        print("Time for 'take unique', 'compute pseudo_action', 'compute reward': " + str(timer))
        print("# of unique pseudo_action for all ref: ", sample_times)

    # compute_score(seq, gts, batch_size, seq_per_img)
    gars = gars.cuda()
    gars = gars.mean(dim=0)
    loss = -(gars[:num_states - 1] * seqlogits[:num_states - 1]).sum() / eff_length
    
    return loss

def pseudo_action_multiple_j(logits, R_cat, pi, temperature=1):
    num_ref = R_cat.size(0)
    maximum_num = 20
    if num_ref <= maximum_num:
        return pseudo_action_fun_cuda(logits, R_cat, pi)
    else:
        times = math.ceil(num_ref / maximum_num)
        for i in range(int(times)):
            start = i * maximum_num
            end = min(num_ref, (i+1) * maximum_num)
            partial_ref = R_cat[start : end]

            if i == 0:
                res = pseudo_action_fun_cuda(logits, partial_ref, pi)
            else:
                res = torch.cat((res, pseudo_action_fun_cuda(logits, partial_ref, pi)))
        return res


def pseudo_action_fun_cuda(logits, R_cat, pi, temperature=1):
    #compute the top two values and their indices
    batch_size, vocab_size = logits.size()
    num_ref = R_cat.size(0)

    top2, indices = torch.topk(-(torch.log(pi) - logits), 2, dim=1)
    top2 = -top2
    min_value = top2[:, 0].unsqueeze(1).expand(num_ref, -1, -1)
    sec_min_value = top2[:, 1].unsqueeze(1).expand(num_ref, -1, -1)
    A_cat = indices[:, 0]
    sec_indices = indices[:, 1]

    pseudo_actions = A_cat.unsqueeze(1).repeat(num_ref, vocab_size).reshape(num_ref, -1, vocab_size)
    pseudo_actions_true_move = sec_indices.unsqueeze(1).repeat(num_ref, vocab_size).reshape(num_ref, -1, vocab_size)

    index_batch = torch.arange(batch_size).cuda().long()
    index_vocab = torch.arange(vocab_size).cuda().long()

    changed1 = -logits.expand(num_ref, -1, -1) + torch.log(pi[:, R_cat].transpose(0, 1).reshape(num_ref, -1, 1))
    changed2 = torch.log(pi).expand(num_ref, -1, -1) - logits[:, R_cat].transpose(0, 1).reshape(num_ref, -1, 1)

    pseudo_actions += (changed1 < min_value).long() * (index_vocab - A_cat.unsqueeze(1)).unsqueeze(0)
    pseudo_actions += (changed2 < min_value).long() * (R_cat.unsqueeze(1).expand(num_ref, batch_size) - A_cat).reshape(num_ref, -1, 1)

    pseudo_actions_true_move += (changed1 < sec_min_value).long() * (changed1 < changed2).long() * (index_vocab - sec_indices.unsqueeze(1)).unsqueeze(0)
    pseudo_actions_true_move += (changed2 < sec_min_value).long() * (changed2 < changed1).long() * (R_cat.unsqueeze(1).expand(num_ref, batch_size) - sec_indices).reshape(num_ref, -1, 1)

    index_matrix = torch.zeros_like(pseudo_actions).long()
    index_matrix[:, index_batch, A_cat] = 1
    index_matrix[R_cat.unsqueeze(1).expand(num_ref, batch_size) == A_cat, :] = 1

    pseudo_actions = pseudo_actions + index_matrix * (pseudo_actions_true_move - pseudo_actions)

    return pseudo_actions












