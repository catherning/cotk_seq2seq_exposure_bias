#coding: utf-8

import math
import sys
from random import random

import numpy as np
import torch
from torch import nn
from torch.nn import Parameter
from torch.nn.modules.rnn import GRU
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from utils import (LongTensor, Storage, Tensor, cuda, gumbel_max,
                   gumbel_max_with_mask, zeros)
from utils.gru_helper import (DecoderRNN, F_GRUCell, SingleAttnGRU,
                              generateMask, maskedSoftmax)

# TODO: create a class and another function to choose which decay ?
def inverse_sigmoid_decay(decay_factor, i):
    return decay_factor / (
        decay_factor + math.exp(i / decay_factor))

def linear_decay(epsilon,offset,slope,i):
    return max(epsilon,offset-slope*i)

def exponential_decay(decay_factor,i):
    return pow(decay_factor,i)


class SingleAttnScheduledSamplingGRU(SingleAttnGRU):
    def forward(self, inp, wLinearLayerCallback, h_init=None, mode='max', input_callback=None, no_unk=True, top_k=10):
        """
        inp contains: batch_size, dm, embLayer, embedding, sampling_proba, max_sent_length, post, post_length, resp_length [init_h]
        input_callback(i, embedding):   if you want to change word embedding at pos i, override this function
        nextStep(embedding, flag):  pass embedding to RNN and get gru_h, flag indicates i th sentence is end when flag[i]==1
        wLinearLayerCallback(gru_h): input gru_h and give a probability distribution on vocablist

        output: w_o emb length"""
        nextStep, h_now, context = self.init_forward_all(
            inp.batch_size, inp.post, inp.post_length, h_init=inp.get("init_h", None))
        
        seqlen = inp.embedding.shape[0]
        start_id = inp.dm.go_id if no_unk else 0
        first_emb = inp.embLayer(LongTensor(
            [inp.dm.go_id])).repeat(inp.batch_size, 1)
        next_emb = first_emb

        gen = Storage()
        gen.w_pro = []
        flag = zeros(inp.batch_size).int()
        EOSmet = []
        length = inp.resp_length - 1

        if input_callback:
            inp.embedding = input_callback(inp.embedding)

        for i in range(seqlen):
            proba = random()

            # Sampling
            if proba < inp.sampling_proba:
                now = next_emb
                if input_callback:
                    now = input_callback(now)

                gru_h = nextStep(now, flag)
                if isinstance(gru_h, tuple):
                    gru_h = gru_h[0]

            # Teacher forcing at this step
            else:
                try:
                    now = inp.embedding[i]
                except IndexError:
                    break

                if self.gru_input_attn:
                    h_now = self.cell_forward(torch.cat([now, context], last_dim=-1), h_now) \
                        * Tensor((length > np.ones(inp.batch_size) * i).astype(float)).unsqueeze(-1)
                else:
                    h_now = self.cell_forward(now, h_now) \
                        * Tensor((length > np.ones(inp.batch_size) * i).astype(float)).unsqueeze(-1)

                query = self.attn_query(h_now)
                attn_weight = maskedSoftmax(
                    (query.unsqueeze(0) * inp.post).sum(-1), inp.post_length)
                context = (attn_weight.unsqueeze(-1) * inp.post).sum(0)
                
                gru_h = torch.cat([h_now, context], dim=-1)

            w = wLinearLayerCallback(gru_h)
            gen.w_pro.append(w)
            
            # Decoding 
            if mode == "max":
                w = torch.argmax(w[:, start_id:], dim=1) + start_id
                next_emb = inp.embLayer(w)
            elif mode == "gumbel" or mode == "sample":
                w_onehot = gumbel_max(w[:, start_id:])
                w = torch.argmax(w_onehot, dim=1) + start_id
                next_emb = torch.sum(torch.unsqueeze(
                    w_onehot, -1) * inp.embLayer.weight[start_id:], 1)
            elif mode == "samplek":
                _, index = w[:, start_id:].topk(
                    top_k, dim=-1, largest=True, sorted=True)  # batch_size, top_k
                mask = torch.zeros_like(
                    w[:, start_id:]).scatter_(-1, index, 1.0)
                w_onehot = gumbel_max_with_mask(w[:, start_id:], mask)
                w = torch.argmax(w_onehot, dim=1) + start_id
                next_emb = torch.sum(torch.unsqueeze(
                    w_onehot, -1) * inp.embLayer.weight[start_id:], 1)
            else:
                raise AttributeError("The given mode {} is not recognized.".format(mode))

            EOSmet.append(flag)
            flag = flag | (w == inp.dm.eos_id).int()
            # The second condition forces the generation (of pad/eos tokens ?) until the generated sentences have a length above resp length
            # In order to be able to calculate the loss. We know the following tokens are pad/eos, but we wouldn't know the proba
            if torch.sum(flag).detach().cpu().numpy() == inp.batch_size and i > inp.embedding.shape[0]:
                break

        EOSmet = 1-torch.stack(EOSmet)
        gen.length = torch.sum(EOSmet, 0).detach().cpu().numpy()
        gen.w_pro = torch.stack(gen.w_pro, dim=0)

        return gen

    def init_forward_all(self, batch_size, post, post_length, h_init=None):
        if h_init is None:
            h_init = self.getInitialParameter(batch_size)
        else:
            h_init = torch.unsqueeze(h_init, 0)
        h_now = h_init[0]
        context = zeros(batch_size, self.post_size)

        def nextStep(incoming, stopmask):
            nonlocal h_now, post, post_length, context

            if self.gru_input_attn:
                h_now = self.cell_forward(torch.cat([incoming, context], dim=-1), h_now) \
                    * (1 - stopmask).float().unsqueeze(-1)
            else:
                h_now = self.cell_forward(
                    incoming, h_now) * (1 - stopmask).float().unsqueeze(-1)

            query = self.attn_query(h_now)
            attn_weight = maskedSoftmax(
                (query.unsqueeze(0) * post).sum(-1), post_length)
            context = (attn_weight.unsqueeze(-1) * post).sum(0)

            return torch.cat([h_now, context], dim=-1), attn_weight 

        return nextStep, h_now, context
