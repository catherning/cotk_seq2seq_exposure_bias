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
	
	def train(self, batch_num):
		args = self.param.args
		dm = self.param.volatile.dm
		datakey = 'train'

		for i in range(batch_num):
			self.now_batch += 1
			incoming = self.get_next_batch(dm, datakey)
			incoming.args = Storage()
			incoming.now_epoch = self.now_epoch

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

	def evaluate(self, key):
		args = self.param.args
		dm = self.param.volatile.dm

		dm.restart(key, args.batch_size, shuffle=False)

		result_arr = []
		while True:
			incoming = self.get_next_batch(dm, key, restart=False)
			if incoming is None:
				break
			incoming.args = Storage()
			incoming.now_epoch = -1
			with torch.no_grad():
				self.net.forward(incoming)
			result_arr.append(incoming.result)

		detail_arr = Storage()
		for i in args.show_sample:
			index = [i * args.batch_size + j for j in range(args.batch_size)]
			incoming = self.get_select_batch(dm, key, index)
			incoming.args = Storage()
			with torch.no_grad():
				self.net.detail_forward(incoming)
			detail_arr["show_str%d" % i] = incoming.result.show_str

		detail_arr.update({key:get_mean(result_arr, key) for key in result_arr[0]})
		detail_arr.perplexity_avg_on_batch = np.exp(detail_arr.word_loss)
		return detail_arr

	def test(self, key):
		args = self.param.args
		dm = self.param.volatile.dm

		metric1 = dm.get_teacher_forcing_metric()
		batch_num, batches = self.get_batches(dm, key)
		logging.info("eval teacher-forcing")
		for incoming in tqdm.tqdm(batches, total=batch_num):
			incoming.args = Storage()
			incoming.now_epoch = -1
			with torch.no_grad():
				self.net.forward(incoming)
				gen_log_prob = nn.functional.log_softmax(incoming.gen.w, -1)
			data = incoming.data
			data.resp_allvocabs = LongTensor(incoming.data.resp_allvocabs)
			data.resp_length = incoming.data.resp_length
			data.gen_log_prob = gen_log_prob.transpose(1, 0)
			metric1.forward(data)
		res = metric1.close()

		metric2 = dm.get_inference_metric()
		batch_num, batches = self.get_batches(dm, key)
		logging.info("eval free-run")
		for incoming in tqdm.tqdm(batches, total=batch_num):
			incoming.args = Storage()
			with torch.no_grad():
				self.net.detail_forward(incoming)
			data = incoming.data
			data.gen = incoming.gen.w_o.detach().cpu().numpy().transpose(1, 0)
			metric2.forward(data)
		res.update(metric2.close())

		if not os.path.exists(args.out_dir):
			os.makedirs(args.out_dir)
		filename = args.out_dir + "/%s_%s.txt" % (args.name, key)

		with codecs.open(filename, 'w',encoding='utf8') as f:
			logging.info("%s Test Result:", key)
			for key, value in res.items():
				if isinstance(value, float) or isinstance(value, str):
					logging.info("\t{}:\t{}".format(key, value))
					f.write("{}:\t{}\n".format(key, value))
			for i in range(len(res['post'])):
				f.write("post:\t%s\n" % " ".join(res['post'][i]))
				f.write("resp:\t%s\n" % " ".join(res['resp'][i]))
				f.write("gen:\t%s\n" % " ".join(res['gen'][i]))
			f.flush()
		logging.info("result output to %s.", filename)
		return {key: val for key, val in res.items() if isinstance(val, (str, int, float))}