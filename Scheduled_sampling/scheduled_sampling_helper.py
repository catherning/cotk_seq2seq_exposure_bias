#coding: utf-8

import sys
sys.path.insert(0, "D:\\Documents\\THU\\Cotk\\cotk_seq2seq_exposure_bias")
import math
from random import random
import numpy as np
import torch
from torch import nn
from torch.nn import Parameter
from torch.nn.modules.rnn import GRU
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from utils import zeros, Tensor, LongTensor, cuda, gumbel_max, gumbel_max_with_mask, Storage
from utils.gru_helper import DecoderRNN, maskedSoftmax, generateMask, F_GRUCell, SingleAttnGRU

def inverse_sigmoid(decay_factor,i):
    return decay_factor / (
            decay_factor + math.exp(i / decay_factor))


class SingleAttnScheduledSamplingGRU(SingleAttnGRU):
    # def __init__(self, input_size, hidden_size, post_size, initpara=True, gru_input_attn=False):
    #     super().__init__()

    #     self.input_size, self.hidden_size, self.post_size = \
    #         input_size, hidden_size, post_size
    #     self.gru_input_attn = gru_input_attn

    #     if self.gru_input_attn:
    #         self.GRU = GRU(input_size + post_size, hidden_size, 1)
    #     else:
    #         self.GRU = GRU(input_size, hidden_size, 1)

    #     self.attn_query = nn.Linear(hidden_size, post_size)

    #     if initpara:
    #         self.h_init = Parameter(torch.Tensor(1, 1, hidden_size))
    #         stdv = 1.0 / math.sqrt(self.hidden_size)
    #         self.h_init.data.uniform_(-stdv, stdv)

    def forward(self, inp, wLinearLayerCallback, h_init=None, mode='max', input_callback=None, no_unk=True, top_k=10):
        """
        inp contains: batch_size, dm, embLayer, max_sent_length, [init_h]
        input_callback(i, embedding):   if you want to change word embedding at pos i, override this function
        nextStep(embedding, flag):  pass embedding to RNN and get gru_h, flag indicates i th sentence is end when flag[i]==1
        wLinearLayerCallback(gru_h): input gru_h and give a probability distribution on vocablist

        output: w_o emb length"""
        # TODO: clean code, adapt nextStep function so that it can be used in teacher forcing?
        # simplifies the sampling mode conditions (repetitive code in sampling & teacher forcing modes)

        nextStep,h_now, context = self.init_forward_all(inp.batch_size, inp.post, inp.post_length, h_init=inp.get("init_h", None))

        start_id = inp.dm.go_id if no_unk else 0
        first_emb = inp.embLayer(LongTensor([inp.dm.go_id])).repeat(inp.batch_size, 1)
        next_emb = first_emb

        gen = Storage()
        gen.w_pro = []
        flag = zeros(inp.batch_size).int()
        EOSmet = []
        length = inp.resp_length -1
        first_time = True

        for i in range(inp.max_sent_length):
            proba = random()
            
            # Sampling
            if proba < inp.sampling_proba:
                now = next_emb  
                first_time = False
                if input_callback:
                    now = input_callback(now)

                gru_h = nextStep(now, flag) # TODO: check, might have to give now[i], like in teacher forcing where we give incoming[i]
                if isinstance(gru_h, tuple):
                    gru_h = gru_h[0]

            
            # Teacher forcing at this step
            else:
                # take min with shape of embed because embedding doesn't have shape max_sent_length, we didn't pad it
                try:
                    now = inp.embedding[i]
                except IndexError:
                    # XXX: if only gen pad tok or sth like that, might be because of that ?
                    now = inp.embLayer(LongTensor([inp.dm.pad_id])).repeat(inp.batch_size, 1) 

                # TODO: check, it was done in function above in teacher forcing, so as if done for all steps at once ?
                if input_callback and first_time:
                    now = input_callback(now)
                    first_time = False

                if self.gru_input_attn:
                    h_now = self.cell_forward(torch.cat([now, context], last_dim=-1), h_now) \
                    * Tensor((length > np.ones(inp.batch_size) * i).astype(float)).unsqueeze(-1)
                else:
                    h_now = self.cell_forward(now, h_now) \
                        * Tensor((length > np.ones(inp.batch_size) * i).astype(float)).unsqueeze(-1)

                query = self.attn_query(h_now)
                attn_weight = maskedSoftmax((query.unsqueeze(0) * inp.post).sum(-1), inp.post_length)
                context = (attn_weight.unsqueeze(-1) * inp.post).sum(0)

            w = wLinearLayerCallback(torch.cat([h_now, context], dim=-1))
            gen.w_pro.append(w)#.softmax(dim=-1))

            if mode == "max":
                w = torch.argmax(w[:, start_id:], dim=1) + start_id
                next_emb = inp.embLayer(w)
            elif mode == "gumbel" or mode == "sample":
                w_onehot = gumbel_max(w[:, start_id:])
                w = torch.argmax(w_onehot, dim=1) + start_id
                next_emb = torch.sum(torch.unsqueeze(w_onehot, -1) * inp.embLayer.weight[start_id:], 1)
            elif mode == "samplek":
                _, index = w[:, start_id:].topk(top_k, dim=-1, largest=True, sorted=True) # batch_size, top_k
                mask = torch.zeros_like(w[:, start_id:]).scatter_(-1, index, 1.0)
                w_onehot = gumbel_max_with_mask(w[:, start_id:], mask)
                w = torch.argmax(w_onehot, dim=1) + start_id
                next_emb = torch.sum(torch.unsqueeze(w_onehot, -1) * inp.embLayer.weight[start_id:], 1)

            EOSmet.append(flag)
            flag = flag | (w == inp.dm.eos_id).int()
            # The second condition forces the generation (of pad/eos tokens ?) until the generated sentences have a length above resp length
            # In order to be able to calculate the loss
            # We know the following tokens are pad/eos, but we wouldn't know the proba
            if torch.sum(flag).detach().cpu().numpy() == inp.batch_size and i>inp.embedding.shape[0]:
                break
        
        EOSmet = 1-torch.stack(EOSmet)
        gen.w_pro = torch.stack(gen.w_pro, dim=0)
        gen.length = torch.sum(EOSmet, 0).detach().cpu().numpy()

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
                h_now = self.cell_forward(incoming, h_now) * (1 - stopmask).float().unsqueeze(-1)

            query = self.attn_query(h_now)
            attn_weight = maskedSoftmax((query.unsqueeze(0) * post).sum(-1), post_length)
            context = (attn_weight.unsqueeze(-1) * post).sum(0)

            return torch.cat([h_now, context], dim=-1), attn_weight

        return nextStep, h_now, context