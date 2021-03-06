# coding:utf-8

import torch

from utils import Storage, flattenSequence
from baseline.network import Network, GenNetwork, EmbeddingLayer, PostEncoder, ConnectLayer
from utils.raml_helper import raml_loss

# pylint: disable=W0221


class RAMLNetwork(Network):
    def __init__(self, param):
        super().__init__(param)
        self.genNetwork = RAMLGenNetwork(param)


class RAMLGenNetwork(GenNetwork):

    def forward(self, incoming):
        inp = Storage()
        inp.resp_length = incoming.data.resp_length
        inp.embedding = incoming.resp.embedding
        inp.post = incoming.hidden.h
        inp.post_length = incoming.data.post_length
        inp.init_h = incoming.conn.init_h

        incoming.gen = gen = Storage()
        self.teacherForcing(inp, gen)

        if self.training:
            incoming.result.word_loss = raml_loss(
                gen.w, incoming.data.resp[1:], incoming.data.resp_length - 1, incoming.data.rewards_ts, self.lossCE)
        else:
            w_o_f = flattenSequence(gen.w, incoming.data.resp_length - 1)
            data_f = flattenSequence(
                incoming.data.resp[1:], incoming.data.resp_length - 1)
            incoming.result.word_loss = self.lossCE(w_o_f, data_f)

        incoming.result.perplexity = torch.exp(incoming.result.word_loss)
