# coding:utf-8
import logging
import os
import time

import numpy as np
import torch
import tqdm
from torch import nn, optim

from baseline.seq2seq import Seq2seq
from .network_raml import RAMLNetwork
from utils import (BaseModel, CheckpointManager, LongTensor, Storage,
                   SummaryHelper, cuda, get_mean, storage_to_list)


class Seq2seqRAML(Seq2seq):
    def __init__(self, param):
        args = param.args
        net = RAMLNetwork(param)
        self.optimizer = optim.Adam(net.get_parameters_by_name(), lr=args.lr)
        optimizerList = {"optimizer": self.optimizer}
        checkpoint_manager = CheckpointManager(args.name, args.model_dir, \
                        args.checkpoint_steps, args.checkpoint_max_to_keep, "min")
        super(Seq2seq,self).__init__(param, net, optimizerList, checkpoint_manager)

        self.create_summary()

    def get_next_batch(self, dm, key, restart=True):
        data = dm.get_next_batch(key)
        if data is None:
            # XXX: might not work cos for now, 2 dm, if raml, then might always get sth, so data never none ?
            print("data batch is none during {}".format(key))
            if restart:
                if key == "train":
                    dm.restart(key, self.args.batch_size // self.args.n_samples)
                else:
                    dm.restart(key)
                return self.get_next_batch(dm, key, False)
            else:
                return None

        return self._preprocess_batch(data)

    def train_process(self):
        """
        The whole training process, all epochs
        """
        args = self.param.args
        dm = self.param.volatile.dm

        while self.now_epoch < args.epochs:
            self.now_epoch += 1
            self.updateOtherWeights()

            dm.restart('train', args.batch_size // args.n_samples)
            self.net.train()
            self.train(args.batch_per_epoch)

            self.net.eval()
            devloss_detail = self.evaluate("dev")
            self.devSummary(self.now_batch, devloss_detail)
            logging.info("epoch %d, evaluate dev", self.now_epoch)

            testloss_detail = self.evaluate("test")
            self.testSummary(self.now_batch, testloss_detail)
            logging.info("epoch %d, evaluate test", self.now_epoch)

            self.save_checkpoint(value=devloss_detail.loss.tolist())
