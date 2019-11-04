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

class Seq2seq(BaseModel):
	def __init__(self, param):
		args = param.args
		net = Network(param)
		self.optimizer = optim.Adam(net.get_parameters_by_name(), lr=args.lr)
		optimizerList = {"optimizer": self.optimizer}
		checkpoint_manager = CheckpointManager(args.name, args.model_dir, \
						args.checkpoint_steps, args.checkpoint_max_to_keep, "min")
		super().__init__(param, net, optimizerList, checkpoint_manager)

		self.create_summary()

	def create_summary(self):
		args = self.param.args
		self.summaryHelper = SummaryHelper("%s/%s_%s" % \
				(args.log_dir, args.name, time.strftime("%H%M%S", time.localtime())), \
				args)
		self.trainSummary = self.summaryHelper.addGroup(\
			scalar=["loss", "word_loss", "perplexity"],\
			prefix="train")

		scalarlist = ["word_loss", "perplexity_avg_on_batch"]
		tensorlist = []
		textlist = []
		emblist = []
		for i in self.args.show_sample:
			textlist.append("show_str%d" % i)
		self.devSummary = self.summaryHelper.addGroup(\
			scalar=scalarlist,\
			tensor=tensorlist,\
			text=textlist,\
			embedding=emblist,\
			prefix="dev")
		self.testSummary = self.summaryHelper.addGroup(\
			scalar=scalarlist,\
			tensor=tensorlist,\
			text=textlist,\
			embedding=emblist,\
			prefix="test")

	def _preprocess_batch(self, data):
		incoming = Storage()
		incoming.data = data = Storage(data)
		data.batch_size = data.post.shape[0]
		data.post = cuda(torch.LongTensor(data.post.transpose(1, 0))) # length * batch_size
		data.resp = cuda(torch.LongTensor(data.resp.transpose(1, 0))) # length * batch_size
		return incoming

	def get_next_batch(self, dm, key, restart=True):
		data = dm.get_next_batch(key)
		if data is None:
			if restart:
				dm.restart(key)
				return self.get_next_batch(dm, key, False)
			else:
				return None
		return self._preprocess_batch(data)

	def get_batches(self, dm, key):
		batches = list(dm.get_batches(key, batch_size=self.args.batch_size, shuffle=False))
		return len(batches), (self._preprocess_batch(data) for data in batches)

	def get_select_batch(self, dm, key, i):
		data = dm.get_batch(key, i)
		if data is None:
			return None
		return self._preprocess_batch(data)

	# TODO: fuse code
	def train(self, batch_num):
		args = self.param.args

		def _train_epoch(sess, epoch_no):
			data_iterator.switch_to_train_data(sess)
			training_log_file = \
				open(log_dir + 'training_log' + str(epoch_no) + '.txt', 'w',
					encoding='utf-8')

			step = 0
			# XXX: for raml
			source_buffer, target_buffer = [], []
			random.shuffle(raml_train_data)
			for training_pair in raml_train_data:
				for target in training_pair['targets']:
					source_buffer.append(training_pair['source'])
					target_buffer.append(target)

				if len(target_buffer) != train_data.batch_size:
					continue

				source_ids = []
				source_length = []
				target_ids = []
				target_length = []
				scores = []

				trunc_len_src = train_data.hparams.source_dataset.max_seq_length
				trunc_len_tgt = train_data.hparams.target_dataset.max_seq_length

				for sentence in source_buffer:
					ids = [train_data.source_vocab.token_to_id_map_py[token]
						for token in sentence.split()][:trunc_len_src]
					ids = ids + [train_data.source_vocab.eos_token_id]

					source_ids.append(ids)
					source_length.append(len(ids))

				for sentence, score_str in target_buffer:
					ids = [train_data.target_vocab.bos_token_id]
					ids = ids + [train_data.target_vocab.token_to_id_map_py[token]
								for token in sentence.split()][:trunc_len_tgt]
					ids = ids + [train_data.target_vocab.eos_token_id]

					target_ids.append(ids)
					scores.append(eval(score_str))
					target_length.append(len(ids))

				rewards = []
				for i in range(0, train_data.batch_size, FLAGS.n_samples):
					tmp = np.array(scores[i:i + FLAGS.n_samples])
					tmp = np.exp(tmp / FLAGS.tau) / np.sum(np.exp(tmp / FLAGS.tau))
					for j in range(0, FLAGS.n_samples):
						rewards.append(tmp[j])

				for value in source_ids:
					while len(value) < max(source_length):
						value.append(0)
				for value in target_ids:
					while len(value) < max(target_length):
						value.append(0)

				feed_dict = {
					batch['source_text_ids']: np.array(source_ids),
					batch['target_text_ids']: np.array(target_ids),
					batch['source_length']: np.array(source_length),
					batch['target_length']: np.array(target_length),
					rewards_ts: np.array(rewards)
				}
				source_buffer = []
				target_buffer = []
				# XXX: to here for raml
				loss = sess.run(train_op, feed_dict=feed_dict)
				print("step={}, loss={:.4f}".format(step, loss),
					file=training_log_file)
				if step % config_data.observe_steps == 0:
					print("step={}, loss={:.4f}".format(step, loss))
				training_log_file.flush()
				step += 1

		dm = self.param.volatile.dm
		datakey = 'train'

		for i in range(batch_num):
			self.now_batch += 1
			incoming = self.get_next_batch(dm, datakey)
			incoming.args = Storage()

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

	def train_process(self):
		"""
		The whole training process, all epochs
		"""
		args = self.param.args
		dm = self.param.volatile.dm

		# TODO: check if here for calling data is ok
		raml_train_data = read_raml_sample_file(args)

		while self.now_epoch < args.epochs:
			self.now_epoch += 1
			self.updateOtherWeights()

			dm.restart('train', args.batch_size)
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

	def test(self, key):
		args = self.param.args
		dm = self.param.volatile.dm

		metric1 = dm.get_teacher_forcing_metric()
		batch_num, batches = self.get_batches(dm, key)
		logging.info("eval teacher-forcing")
		for incoming in tqdm.tqdm(batches, total=batch_num):
			incoming.args = Storage()
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

		with open(filename, 'w') as f:
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

	def test_process(self):
		logging.info("Test Start.")
		self.net.eval()
		self.test("dev")
		test_res = self.test("test")
		logging.info("Test Finish.")
		return test_res
