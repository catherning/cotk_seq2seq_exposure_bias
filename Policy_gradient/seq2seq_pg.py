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

# from network import Network
from Policy_gradient.network_pg import NetworkPG


from baseline.seq2seq import Seq2seq

# Nouveau fichier
# policygradient
class Seq2seqPG(Seq2seq):
	def __init__(self, param):
		args = param.args
		net = NetworkPG(param)
		self.optimizer = optim.Adam(net.get_parameters_by_name(), lr=args.lr)
		optimizerList = {"optimizer": self.optimizer}
		checkpoint_manager = CheckpointManager(args.name, args.model_dir, \
						args.checkpoint_steps, args.checkpoint_max_to_keep, "min")
		super(Seq2seq, self).__init__(param, net, optimizerList, checkpoint_manager)

		self.create_summary()