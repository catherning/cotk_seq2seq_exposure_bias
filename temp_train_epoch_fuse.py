# coding:utf-8
import logging
import time
import os

import torch
from torch import nn, optim
import numpy as np
import tqdm

import random

from utils import Storage, cuda, BaseModel, SummaryHelper, get_mean, storage_to_list, \
	CheckpointManager, LongTensor, read_raml_sample_file
from network import Network

# TODO: put that in new dataloader
# for dm => convert to raml file
# train_data is dm


# TODO: create new dataloader! for rewards
# from textprocessing dataloader. To take into account the nb of samples
# def get_next_batch(ignore_left_samples=False):
# 		'''Get next batch. It can be called only after Initializing batches (:func:`restart`).

# 	Arguments:
# 		key (str): key name of dataset, must be contained in ``self.key_name``.
# 		ignore_left_samples (bool): If the number of left samples is not equal to
# 			``batch_size``, ignore them. This make sure all batches have same number of samples.
# 			Default: ``False``

# 	Returns:
# 		A dict like :func:`get_batch`, or None if the epoch is end.
# 	'''
# 	if key not in self.key_name:
#     		raise ValueError("No set named %s." % key)
# 	if self.batch_size[key] is None:
# 		raise RuntimeError( \
# 			"Please run restart before calling this function.")
# 	batch_id = self.batch_id[key]
# 	start, end = batch_id * \
# 		self.batch_size[key], (batch_id + 1) * self.batch_size[key]
# 	if start >= len(self.index[key]):
# 		return None
# 	if ignore_left_samples and end > len(self.index[key]):
# 		return None
# 	index = self.index[key][start:end]
# 	res = self.get_batch(key, index)
# 	self.batch_id[key] += 1
# 	return res