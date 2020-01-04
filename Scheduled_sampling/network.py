# coding:utf-8
import logging

import torch
from torch import nn

from utils import zeros, LongTensor,\
            BaseNetwork, MyGRU, Storage, gumbel_max, flattenSequence, SingleAttnGRU, SequenceBatchNorm
from baseline.network import Network,GenNetwork
from scheduled_sampling_helper import SingleAttnScheduledSamplingGRU

# pylint: disable=W0221
class ScheduledSamplingNetwork(Network):
    def __init__(self, param):
        super().__init__(param)

        self.genNetwork = ScheduledSamplingGenNetwork(param)


class ScheduledSamplingGenNetwork(GenNetwork):
    def __init__(self, param):
        super().__init__(param)
        args = param.args

        self.GRULayer = SingleAttnScheduledSamplingGRU(args.embedding_size, args.dh_size, args.eh_size * 2, initpara=False)

    # TODO: create sampling proba def or attribute that get updated, so that i don't recreate test & eval functions in seq2seq ?

    def scheduledTeacherForcing(self, inp, gen):
        def input_callback(now):
            return self.drop(now)

        def wLinearLayerCallback(gru_h):
            gru_h = self.drop(gru_h)
            w = self.wLinearLayer(gru_h)
            return w

        # for now, will NOT accept beam mode
        new_gen = self.GRULayer.forward(inp, wLinearLayerCallback, mode=self.args.decode_mode, input_callback=input_callback, h_init=inp.init_h)
        gen.length = new_gen.length
        gen.w_pro = new_gen.w_pro

    def forward(self, incoming):
        # TODO: call this function
        inp = Storage()
        inp.embedding = incoming.resp.embedding
        inp.post = incoming.hidden.h
        inp.post_length = incoming.data.post_length
        inp.resp_length = incoming.data.resp_length
        incoming.gen = gen = Storage()
        inp.init_h = incoming.conn.init_h
        
        # if self.training:
        inp.embLayer = incoming.resp.embLayer
        inp.max_sent_length = self.args.max_sent_length
        inp.sampling_proba = incoming.args.sampling_proba
        inp.dm = self.param.volatile.dm
        inp.batch_size = incoming.data.batch_size

        self.scheduledTeacherForcing(inp, gen)
    
        # else:
        #     self.teacherForcing(inp, gen)

        w_o_f = flattenSequence(gen.w_pro, incoming.data.resp_length-1)
        data_f = flattenSequence(incoming.data.resp[1:], incoming.data.resp_length-1)
        incoming.result.word_loss = self.lossCE(w_o_f, data_f)
        incoming.result.perplexity = torch.exp(incoming.result.word_loss)
