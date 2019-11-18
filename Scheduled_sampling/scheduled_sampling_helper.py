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
from utils.gru_helper import DecoderRNN, maskedSoftmax, generateMask, F_GRUCell

def inverse_sigmoid(decay_factor,i):
    return decay_factor / (
            decay_factor + math.exp(i / decay_factor))


class SingleAttnScheduledSamplingGRU(DecoderRNN):
    def __init__(self, input_size, hidden_size, post_size, initpara=True, gru_input_attn=False):
        super().__init__()

        self.input_size, self.hidden_size, self.post_size = \
            input_size, hidden_size, post_size
        self.gru_input_attn = gru_input_attn

        if self.gru_input_attn:
            self.GRU = GRU(input_size + post_size, hidden_size, 1)
        else:
            self.GRU = GRU(input_size, hidden_size, 1)

        self.attn_query = nn.Linear(hidden_size, post_size)

        if initpara:
            self.h_init = Parameter(torch.Tensor(1, 1, hidden_size))
            stdv = 1.0 / math.sqrt(self.hidden_size)
            self.h_init.data.uniform_(-stdv, stdv)

    def getInitialParameter(self, batch_size):
        return self.h_init.repeat(1, batch_size, 1)

    # def forward(self, incoming, length, post, post_length, h_init=None):
    #     """
    #     Original forward
    #     """
    #     batch_size = incoming.shape[1]
    #     seqlen = incoming.shape[0]
    #     if h_init is None:
    #         h_init = self.getInitialParameter(batch_size)
    #     else:
    #         h_init = torch.unsqueeze(h_init, 0)
    #     h_now = h_init[0]
    #     hs = []
    #     attn_weights = []
    #     context = zeros(batch_size, self.post_size)

    #     for i in range(seqlen):
    #         if self.gru_input_attn:
    #             h_now = self.cell_forward(torch.cat([incoming[i], context], last_dim=-1), h_now) \
    #                 * Tensor((length > np.ones(batch_size) * i).astype(float)).unsqueeze(-1)
    #         else:
    #             h_now = self.cell_forward(incoming[i], h_now) \
    #                 * Tensor((length > np.ones(batch_size) * i).astype(float)).unsqueeze(-1)

    #         query = self.attn_query(h_now)
    #         attn_weight = maskedSoftmax((query.unsqueeze(0) * post).sum(-1), post_length)
    #         context = (attn_weight.unsqueeze(-1) * post).sum(0)

    #         hs.append(torch.cat([h_now, context], dim=-1))
    #         attn_weights.append(attn_weight)

    #     return h_now, hs, attn_weights

    def new_forward(self, inp, wLinearLayerCallback, h_init=None, mode='max', input_callback=None, no_unk=True, top_k=10):
        """
        inp contains: batch_size, dm, embLayer, max_sent_length, [init_h]
        input_callback(i, embedding):   if you want to change word embedding at pos i, override this function
        nextStep(embedding, flag):  pass embedding to RNN and get gru_h, flag indicates i th sentence is end when flag[i]==1
        wLinearLayerCallback(gru_h): input gru_h and give a probability distribution on vocablist

        output: w_o emb length"""

        # TODO: check for teacher forcing, the resp is of size max(lengths) or max_length = seqlength= 50 ? 
        # because if y_(t-1) index out of range if above max(lengths) for a certain batch
        # TODO: do something about seqlen = incoming.shape[0] ? Genre if i ==seqlen in the main loop, then everything else si 0 ? 
        # Or should learn by itself, that's the goal of freerun!

        # XXX: could give that as argument eventually
        batch_size = inp.batch_size
        nextStep,h_now, context = self.init_forward(inp.batch_size, inp.post, inp.post_length, inp.get("init_h", None))

        start_id = inp.dm.go_id if no_unk else 0
        dm = inp.dm
        first_emb = inp.embLayer(LongTensor([dm.go_id])).repeat(batch_size, 1)
        next_emb = first_emb

        gen = Storage()
        gen.w_pro = []
        gen.w_o = []
        gen.emb = []
        flag = zeros(batch_size).int()
        EOSmet = []

        # forward init => init in init_forward which is in nextStep
        # if h_init is None:
        #     h_init = self.getInitialParameter(batch_size)
        # else:
        #     h_init = torch.unsqueeze(h_init, 0)
        # h_now = h_init[0]
        # context = zeros(batch_size, self.post_size)

        # these useless because hs needed when didn't use wLinearLayerCallback, attn_weights not used ?
        # hs = []
        # attn_weights = []
        length = inp.resp_length -1

        # TODO: we use input_callback only once at the beginning for teacher forcing, but each time when freerun.
        # => do it once only if first step is teacher forcing and then each time when freerun, 
        # otherwise, for the other teacher forcing steps, we don't do it
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

                w = wLinearLayerCallback(gru_h)
                gen.w_pro.append(w.softmax(dim=-1))

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

                gen.w_o.append(w)
                gen.emb.append(next_emb)
            
            # Teacher forcing at this step
            else:
                now = inp.embedding
                # TODO: check, it was done in function above in teacher forcing, so as if done for all steps at once ?
                if input_callback and first_time:
                    now = input_callback(now)
                    first_time = False
                now = now[i]

                if self.gru_input_attn:
                    h_now = self.cell_forward(torch.cat([now, context], last_dim=-1), h_now) \
                    * Tensor((length > np.ones(batch_size) * i).astype(float)).unsqueeze(-1)
                else:
                    h_now = self.cell_forward(now, h_now) \
                        * Tensor((length > np.ones(batch_size) * i).astype(float)).unsqueeze(-1)

                query = self.attn_query(h_now)
                attn_weight = maskedSoftmax((query.unsqueeze(0) * inp.post).sum(-1), inp.post_length)
                context = (attn_weight.unsqueeze(-1) * inp.post).sum(0)

                w = wLinearLayerCallback(torch.cat([h_now, context], dim=-1))
                gen.w_o.append(w)
                # Randomly took this def from max sampling. TODO: check if ok
                next_emb = inp.embLayer(w) # w is NOT correct

                # attn_weights.append(attn_weight) # XXX: need it ?

                # TODO: recheck where to put that: in or out of the if proba ?
                EOSmet.append(flag)
                flag = flag | (w == dm.eos_id).int()
                if torch.sum(flag).detach().cpu().numpy() == batch_size:
                    break

        EOSmet = 1-torch.stack(EOSmet)
        gen.w_o = torch.stack(gen.w_o) * EOSmet.long()
        gen.emb = torch.stack(gen.emb) * EOSmet.float().unsqueeze(-1)
        gen.length = torch.sum(EOSmet, 0).detach().cpu().numpy()

        return gen

    def init_forward(self, batch_size, post, post_length, h_init=None):
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

    def init_forward_3d(self, batch_size, top_k, post, post_length, h_init=None):
        if h_init is None:
            h_init = self.getInitialParameter(batch_size)
        else:
            h_init = torch.unsqueeze(h_init, 0)
        h_now = h_init[0].unsqueeze(1).expand(-1, top_k, -1) # batch_size * top_k * hidden_size
        context = zeros(batch_size, self.post_size)

        post = post.unsqueeze(-2)
        #post_length = np.tile(np.expand_dims(post_length, 1), (1, top_k, 1))

        def nextStep(incoming, stopmask, regroup=None):
            nonlocal h_now, post, post_length, context
            h_now = torch.gather(h_now, 1, regroup.unsqueeze(-1).repeat(1, 1, h_now.shape[-1]))

            if self.gru_input_attn:
                context = torch.gather(context, 1, regroup.unsqueeze(-1).repeat(1, 1, context.shape[-1]))
                h_now = self.cell_forward(torch.cat([incoming, context], dim=-1), h_now) \
                    * (1 - stopmask).float().unsqueeze(-1)
            else:
                h_now = self.cell_forward(incoming, h_now) * (1 - stopmask).float().unsqueeze(-1) # batch_size * top_k * hidden_size

            query = self.attn_query(h_now) # batch_size * top_k * post_size

            mask = generateMask(post.shape[0], post_length).unsqueeze(-1)
            attn_weight = (query.unsqueeze(0) * post).sum(-1).masked_fill(mask==0, -1e9).softmax(0) # post_len * batch_size * top_k
            context = (attn_weight.unsqueeze(-1) * post).sum(0)

            return torch.cat([h_now, context], dim=-1), attn_weight

        return nextStep

    def cell_forward(self, incoming, h):
        shape = h.shape
        return F_GRUCell( \
                incoming.reshape(-1, incoming.shape[-1]), h.reshape(-1, h.shape[-1]), \
                self.GRU.weight_ih_l0, self.GRU.weight_hh_l0, \
                self.GRU.bias_ih_l0, self.GRU.bias_hh_l0, \
        ).reshape(*shape)

    def freerun(self, inp, wLinearLayerCallback, mode='max', input_callback=None, no_unk=True, top_k=10):
        nextStep = self.init_forward(inp.batch_size, inp.post, inp.post_length, inp.get("init_h", None))
        return self._freerun(inp, nextStep, wLinearLayerCallback, mode, input_callback, no_unk, top_k=top_k)

    def beamsearch(self, inp, top_k, wLinearLayerCallback, input_callback=None, no_unk=True, length_penalty=0.7):
        nextStep = self.init_forward_3d(inp.batch_size, top_k, inp.post, inp.post_length, inp.get("init_h", None))
        return self._beamsearch(inp, top_k, nextStep, wLinearLayerCallback, input_callback, no_unk, length_penalty)
