# coding:utf-8
import logging
import time
import os

import torch
from torch import nn, optim
import numpy as np
import tqdm

from utils import Storage, cuda, BaseModel, SummaryHelper, get_mean, storage_to_list, \
    CheckpointManager, LongTensor
from Scheduled_sampling.scheduled_sampling_helper import inverse_sigmoid
from network import ScheduledSamplingNetwork
from baselines.cotk_seq2seq_code.seq2seq import Seq2seq

class ScheduledSamplingSeq2seq(Seq2seq):
    def __init__(self, param):
        args = param.args
        net = ScheduledSamplingNetwork(param)
        self.optimizer = optim.Adam(net.get_parameters_by_name(), lr=args.lr)
        optimizerList = {"optimizer": self.optimizer}
        checkpoint_manager = CheckpointManager(args.name, args.model_dir, \
                        args.checkpoint_steps, args.checkpoint_max_to_keep, "min")
        super(Seq2seq,self).__init__(param, net, optimizerList, checkpoint_manager)

        self.create_summary()

    def train(self, batch_num, total_step_counter):
        args = self.param.args
        dm = self.param.volatile.dm
        datakey = 'train'

        for i in range(batch_num):
            self.now_batch += 1
            incoming = self.get_next_batch(dm, datakey)
            incoming.args = Storage()
            incoming.args.sampling_proba = 1. - inverse_sigmoid(args.decay_factor,total_step_counter)

            if (i+1) % args.batch_num_per_gradient == 0:
                self.zero_grad()
            self.net.forward(incoming)

            loss = incoming.result.loss
            self.trainSummary(self.now_batch, storage_to_list(incoming.result))
            logging.info("batch %d : gen loss=%f", self.now_batch, loss.detach().cpu().numpy())

            loss.backward()

            if (i+1) % args.batch_num_per_gradient == 0:
                nn.utils.clip_grad_norm_(self.net.parameters(), args.grad_clip)
                self.optimizer.step()

            total_step_counter += 1

    def train_process(self):
        args = self.param.args
        dm = self.param.volatile.dm

        total_step_counter = 1
        while self.now_epoch < args.epochs:
            self.now_epoch += 1
            self.updateOtherWeights()

            dm.restart('train', args.batch_size)
            self.net.train()
            self.train(args.batch_per_epoch, total_step_counter)

            self.net.eval()
            devloss_detail = self.evaluate("dev")
            self.devSummary(self.now_batch, devloss_detail)
            logging.info("epoch %d, evaluate dev", self.now_epoch)

            testloss_detail = self.evaluate("test")
            self.testSummary(self.now_batch, testloss_detail)
            logging.info("epoch %d, evaluate test", self.now_epoch)

            self.save_checkpoint(value=devloss_detail.loss.tolist())