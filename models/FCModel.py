from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from multiprocessing import Pool
from torch.autograd import *
import misc.utils as utils

from .CaptionModel import CaptionModel

class LSTMCore(nn.Module):
    def __init__(self, opt):
        super(LSTMCore, self).__init__()
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_size = opt.rnn_size
        self.drop_prob_lm = opt.drop_prob_lm
        
        # Build a LSTM
        self.i2h = nn.Linear(self.input_encoding_size, 5 * self.rnn_size)
        self.h2h = nn.Linear(self.rnn_size, 5 * self.rnn_size)
        self.dropout = nn.Dropout(self.drop_prob_lm)

    def forward(self, xt, state):

        all_input_sums = self.i2h(xt) + self.h2h(state[0][-1])
        sigmoid_chunk = all_input_sums.narrow(1, 0, 3 * self.rnn_size)
        sigmoid_chunk = F.sigmoid(sigmoid_chunk)
        in_gate = sigmoid_chunk.narrow(1, 0, self.rnn_size)
        forget_gate = sigmoid_chunk.narrow(1, self.rnn_size, self.rnn_size)
        out_gate = sigmoid_chunk.narrow(1, self.rnn_size * 2, self.rnn_size)

        in_transform = torch.max(\
            all_input_sums.narrow(1, 3 * self.rnn_size, self.rnn_size),
            all_input_sums.narrow(1, 4 * self.rnn_size, self.rnn_size))
        next_c = forget_gate * state[1][-1] + in_gate * in_transform
        next_h = out_gate * F.tanh(next_c)

        output = self.dropout(next_h)
        state = (next_h.unsqueeze(0), next_c.unsqueeze(0))
        return output, state

class FCModel(CaptionModel):
    def __init__(self, opt):
        super(FCModel, self).__init__()
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length
        self.fc_feat_size = opt.fc_feat_size

        self.ss_prob = 0.0 # Schedule sampling probability

        self.img_embed = nn.Linear(self.fc_feat_size, self.input_encoding_size)
        self.core = LSTMCore(opt)
        self.embed = nn.Embedding(self.vocab_size + 1, self.input_encoding_size)
        self.logit = nn.Linear(self.rnn_size, self.vocab_size + 1)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)
        self.logit.bias.data.fill_(0)
        self.logit.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'lstm':
            return (weight.new_zeros(self.num_layers, bsz, self.rnn_size),
                    weight.new_zeros(self.num_layers, bsz, self.rnn_size))
        else:
            return weight.new_zeros(self.num_layers, bsz, self.rnn_size)

    def _forward(self, fc_feats, att_feats, seq, att_masks=None):
        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)
        outputs = []

        for i in range(seq.size(1)):
            if i == 0:
                xt = self.img_embed(fc_feats)
            else:
                if self.training and i >= 2 and self.ss_prob > 0.0: # otherwiste no need to sample
                    sample_prob = fc_feats.data.new(batch_size).uniform_(0, 1)
                    sample_mask = sample_prob < self.ss_prob
                    if sample_mask.sum() == 0:
                        it = seq[:, i-1].clone()
                    else:
                        sample_ind = sample_mask.nonzero().view(-1)
                        it = seq[:, i-1].data.clone()
                        #prob_prev = torch.exp(outputs[-1].data.index_select(0, sample_ind)) # fetch prev distribution: shape Nx(M+1)
                        #it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1))
                        prob_prev = torch.exp(outputs[-1].data) # fetch prev distribution: shape Nx(M+1)
                        it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
                else:
                    it = seq[:, i-1].clone()
                # break if all the sequences end
                #if i >= 2 and seq[:, i-1].sum() == 0:
                    #break
                xt = self.embed(it)

            output, state = self.core(xt, state)
            output = F.log_softmax(self.logit(output), dim=1)
            outputs.append(output)

        return torch.cat([_.unsqueeze(1) for _ in outputs[1:]], 1).contiguous()

    def get_logprobs_state(self, it, state):
        # 'it' is contains a word index
        xt = self.embed(it)

        output, state = self.core(xt, state)
        logprobs = F.log_softmax(self.logit(output), dim=1)

        return logprobs, state

    def _sample_beam(self, fc_feats, att_feats, att_masks=None, opt={}):
        beam_size = opt.get('beam_size', 10)
        batch_size = fc_feats.size(0)

        assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = self.init_hidden(beam_size)
            for t in range(2):
                if t == 0:
                    xt = self.img_embed(fc_feats[k:k+1]).expand(beam_size, self.input_encoding_size)
                elif t == 1: # input <bos>
                    it = fc_feats.data.new(beam_size).long().zero_()
                    xt = self.embed(it)

                output, state = self.core(xt, state)
                logprobs = F.log_softmax(self.logit(output), dim=1)

            self.done_beams[k] = self.beam_search(state, logprobs, opt=opt)
            seq[:, k] = self.done_beams[k][0]['seq'] # the first beam has highest cumulative score
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']
        # return the samples and their log likelihoods
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)

    def _sample(self, fc_feats, att_feats, att_masks=None, opt={}):
        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        if beam_size > 1:
            return self._sample_beam(fc_feats, att_feats, opt)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)
        seq = fc_feats.new_zeros(batch_size, self.seq_length, dtype=torch.long)
        seqLogprobs = fc_feats.new_zeros(batch_size, self.seq_length)
        for t in range(self.seq_length + 2):
            if t == 0:
                xt = self.img_embed(fc_feats)
            else:
                if t == 1: # input <bos>
                    it = fc_feats.data.new(batch_size).long().zero_()
                xt = self.embed(it)

            output, state = self.core(xt, state)
            logprobs = F.log_softmax(self.logit(output), dim=1)

            # sample the next_word
            if t == self.seq_length + 1: # skip if we achieve maximum length
                break
            if sample_max:
                sampleLogprobs, it = torch.max(logprobs.data, 1)
                it = it.view(-1).long()
            else:
                if temperature == 1.0:
                    prob_prev = torch.exp(logprobs.data).cpu() # fetch prev distribution: shape Nx(M+1)
                else:
                    # scale logprobs by temperature
                    prob_prev = torch.exp(torch.div(logprobs.data, temperature)).cpu()
                it = torch.multinomial(prob_prev, 1).cuda()
                sampleLogprobs = logprobs.gather(1, it) # gather the logprobs at sampled positions
                it = it.view(-1).long() # and flatten indices for downstream processing

            if t >= 1:
                # stop when all finished
                if t == 1:
                    unfinished = it > 0
                else:
                    unfinished = unfinished * (it > 0)
                it = it * unfinished.type_as(it)
                seq[:,t-1] = it #seq[t] the input of t+2 time step
                seqLogprobs[:,t-1] = sampleLogprobs.view(-1)
                if unfinished.sum() == 0:
                    break

        return seq, seqLogprobs


    def _arm_sample(self, fc_feats, att_feats, att_masks=None, opt={}):
        sample_max = opt.get('sample_max', 1)
        temperature = opt.get('temperature', 1.0)
        batch_size = fc_feats.size(0)

        state_list = []

        state = self.init_hidden(batch_size)

        seq = fc_feats.new_zeros(batch_size, self.seq_length, dtype=torch.long)
        seqLogits = fc_feats.new_zeros(self.seq_length, batch_size, self.vocab_size+1)

        for t in range(0, self.seq_length + 1):
            if t == 0:
                xt = self.img_embed(fc_feats)
            else:
                if t == 1: # input <bos>
                    it = fc_feats.data.new(batch_size).long().zero_()
                xt = self.embed(it)

            output, state = self.core(xt, state)
            logits = self.logit(output)
            logprobs = F.log_softmax(logits, dim=1)

            # sample the next_word
            if sample_max:
                sampleLogprobs, it = torch.max(logprobs.data, 1)
                it = it.view(-1).long()
            else:
                if temperature == 1.0:
                    prob_prev = torch.exp(logprobs.data).cpu() # fetch prev distribution: shape Nx(M+1)
                else:
                    # scale logprobs by temperature
                    prob_prev = torch.exp(torch.div(logprobs.data, temperature)).cpu()
                it = torch.multinomial(prob_prev, 1).cuda()
                sampleLogprobs = logprobs.gather(1, it) # gather the logprobs at sampled positions
                it = it.view(-1).long() # and flatten indices for downstream processing

            if t >= 1:
                # stop when all finished
                if t == 1:
                    unfinished = it > 0
                else:
                    unfinished = unfinished * (it > 0)
                it = it * unfinished.type_as(it)
                seq[:, t-1] = it #seq[t] the input of t+2 time step
                seqLogits[t - 1, :, :] = logits

            state_h, state_c = state
            state_list.append((state_h.cpu().data, state_c.cpu().data))

            if t >= 1 and unfinished.sum() == 0:
                break
	
        eff_length = (seq > 0).sum().float()

        return seq, seqLogits, eff_length, state_list


    def _arm_pseudo_sample(self, t_start, it, state, seq, opt={}):
        sample_max = opt.get('sample_max', 1)
        temperature = opt.get('temperature', 1.0)

        it = it.clone()
        state_h, state_c = state
        state = (state_h.clone(), state_c.clone())
        seq = seq.clone()

        # self.core.share_memory()

        # Determines where to start to sample to the end for reward evaluation.

        assert t_start > 0

        # input the previous states and current it, and sample to the end


        for t in range(t_start, self.seq_length + 1):

            if t == t_start:
                unfinished = it > 0
            else:
                unfinished = unfinished * (it > 0)
            it = it * unfinished.type_as(it)
            seq[:, t - 1] = it

            if unfinished.sum() == 0:
                break

            xt = self.embed(it)

            output, state = self.core(xt, state)
            logits = self.logit(output)
            logprobs = F.log_softmax(logits, dim=1)

            # sample the next_word
            if sample_max:
                sampleLogprobs, it = torch.max(logprobs.data, 1)
                it = it.view(-1).long()
            else:
                if temperature == 1.0:
                    prob_prev = torch.exp(logprobs.data).cpu() # fetch prev distribution: shape Nx(M+1)
                else:
                    # scale logprobs by temperature
                    prob_prev = torch.exp(torch.div(logprobs.data, temperature)).cpu()
                it = torch.multinomial(prob_prev, 1).cuda()
                sampleLogprobs = logprobs.gather(1, it) # gather the logprobs at sampled positions
                it = it.view(-1).long() # and flatten indices for downstream processing

        return seq


    # def _arm_pseudo_sample(self, fc_feats, att_feats, att_masks=None, opt={}):
    #     sample_max = opt.get('sample_max', 1)
    #     temperature = opt.get('temperature', 1.0)
    #     batch_size = fc_feats.size(0)
    #
    #     save_state = opt.get('save_state', False)
    #     save_logits = opt.get('save_logits', False)
    #     it_list = []
    #     state_list = []
    #
    #     # Determines where to start to sample to the end for reward evaluation.
    #     t_start = opt.get('t_start', 0)
    #
    #     # If t_start = 0, sample the whole trajectory , and save all states and its for pseudo action sampling.
    #     # If t_start > 0, get the previous sequence information.
    #     if t_start == 0:
    #         state = self.init_hidden(batch_size)
    #
    #         seq = fc_feats.new_zeros(batch_size, self.seq_length, dtype=torch.long)
    #         # seqLogprobs = fc_feats.new_zeros(batch_size, self.seq_length)
    #         seqLogits = fc_feats.new_zeros(self.seq_length, batch_size, self.vocab_size+1)
    #     else:
    #         it = opt.get('pre_it', None)
    #         state = opt.get('pre_state', None)
    #
    #         seq = opt.get('pre_seq', None)
    #         # seqLogprobs = opt.get('pre_seqLogprobs', None)
    #         seqLogits = opt.get('pre_Logits', None)
    #
    #
    #     for t in range(t_start, self.seq_length + 1):
    #         if t == 0:
    #             xt = self.img_embed(fc_feats)
    #         else:
    #             if t == 1: # input <bos>
    #                 it = fc_feats.data.new(batch_size).long().zero_()
    #             xt = self.embed(it)
    #
    #         output, state = self.core(xt, state)
    #         logits = self.logit(output)
    #         logprobs = F.log_softmax(logits, dim=1)
    #
    #         # sample the next_word
    #         if sample_max:
    #             sampleLogprobs, it = torch.max(logprobs.data, 1)
    #             it = it.view(-1).long()
    #         else:
    #             if temperature == 1.0:
    #                 prob_prev = torch.exp(logprobs.data).cpu() # fetch prev distribution: shape Nx(M+1)
    #             else:
    #                 # scale logprobs by temperature
    #                 prob_prev = torch.exp(torch.div(logprobs.data, temperature)).cpu()
    #             it = torch.multinomial(prob_prev, 1).cuda()
    #             sampleLogprobs = logprobs.gather(1, it) # gather the logprobs at sampled positions
    #             it = it.view(-1).long() # and flatten indices for downstream processing
    #
    #         if t >= 1:
    #             # stop when all finished
    #             if t == 1 or t == t_start:
    #                 unfinished = it > 0
    #             else:
    #                 unfinished = unfinished * (it > 0)
    #             it = it * unfinished.type_as(it)
    #             seq[:, t-1] = it #seq[t] the input of t+2 time step
    #             # seqLogprobs[:, t-1] = sampleLogprobs.view(-1)
    #             if save_logits:
    #                 seqLogits[t-1, :, :] = logits
    #
    #         if save_state:
    #             it_list.append(it)
    #             state_list.append(state)
    #
    #         if t >= 1 and unfinished.sum() == 0:
    #             break
    #
    #     if save_state and save_logits:
    #         return seq, seqLogits, it_list, state_list
    #     else:
    #         return seq
