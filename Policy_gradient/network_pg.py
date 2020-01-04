# coding:utf-8
import logging

import torch
from torch import nn


from utils import zeros, LongTensor,\
			BaseNetwork, MyGRU, Storage, gumbel_max, flattenSequence, SingleAttnGRU, SequenceBatchNorm

from utils import LossRL, cuda


from baseline.network import Network, GenNetwork


#policy gradient
class NetworkPG(Network):
	def __init__(self, param):
		super().__init__(param)
		self.genNetwork = GenNetworkPG(param)

class GenNetworkPG(GenNetwork):
	def __init__(self, param):
		super().__init__(param)		
		self.lossPG = LossRL.PolicyGradientLoss()
	
	def samplingLearning(self, inp, gen, mode="gumbel"):
		def wLinearLayerCallback(gru_h):
			gru_h = self.drop(gru_h)
			w = self.wLinearLayer(gru_h)
			return w

		def input_callback(i, now):
			return self.drop(now)

		new_gen = self.GRULayer.freerun(inp, wLinearLayerCallback, mode, \
			input_callback=input_callback, no_unk=True, top_k=self.args.top_k)
		gen.w_o = new_gen.w_o
		gen.w_pro = torch.stack(new_gen.w_pro)
		gen.length = new_gen.length


	def samplingLearningMultiple(self, inp, gen, mode="gumbel"):
		def wLinearLayerCallback(gru_h):
			gru_h = self.drop(gru_h)
			w = self.wLinearLayer(gru_h)
			return w

		def input_callback(i, now):
			return self.drop(now)

		list_wo, list_w_pro, list_length = [],[],[]
		for i in range(inp.nb_sample_training):
			temp_gen = self.GRULayer.freerun(inp, wLinearLayerCallback, mode, \
				input_callback=input_callback, no_unk=True, top_k=self.args.top_k)
			list_wo.append(temp_gen.w_o)
			list_w_pro.append(torch.stack(temp_gen.w_pro))
			list_length.append(cuda(torch.tensor(temp_gen.length)).unsqueeze(0))

		temp_len = [torch.max(x) for x in list_length]
		i_ord = torch.argsort(-torch.tensor(temp_len))

		from torch.nn.utils.rnn import pad_sequence
		import numpy as np

		list_wo_np    = np.array(list_wo)[i_ord] # Only to change the order of the samples in a decreasing order
		list_w_pro_np = np.array(list_w_pro)[i_ord]

		gen.w_o   = pad_sequence(list_wo_np,    batch_first=True) 
		gen.w_pro = pad_sequence(list_w_pro_np, batch_first=True) 

		gen.length = torch.cat(list_length)[i_ord]
#y761GTPT
		# w_o : shape = nb_sample x sentence_size x batch_size 
		# gen.w_pro    = torch.cat(list_wo)
		# gen.w_pro  = torch.cat(list_w_pro)


	def forward(self, incoming):
		"""
		 Ranzatoet  al.[28]  suggested  to  pre-train  
		 the model for a few epochs using the cross-entropy loss 
		 and thens slowly  switch  to  the  REINFORCE  loss
		"""
		
		def teacherForcingForward(incoming): #Copied from GenNetwork.forward
			inp = Storage()
			inp.resp_length = incoming.data.resp_length
			inp.embedding = incoming.resp.embedding
			inp.post = incoming.hidden.h
			inp.post_length = incoming.data.post_length
			inp.init_h = incoming.conn.init_h

			incoming.gen = gen = Storage()

			self.teacherForcing(inp, gen)

			w_o_f = flattenSequence(gen.w, incoming.data.resp_length-1)
			data_f = flattenSequence(incoming.data.resp[1:], incoming.data.resp_length-1)
			
			incoming.result.word_loss = self.lossCE(w_o_f, data_f)
			incoming.result.perplexity = torch.exp(incoming.result.word_loss)
		
		def policyGradientForward(incoming):
			inp = Storage()
			batch_size = inp.batch_size = incoming.data.batch_size####
			inp.init_h = incoming.conn.init_h#
			inp.post = incoming.hidden.h#
			inp.post_length = incoming.data.post_length#
			inp.embLayer = incoming.resp.embLayer####
			inp.dm = self.param.volatile.dm####
			inp.max_sent_length = self.args.max_sent_length ####

			inp.nb_sample_training = self.param.args.nb_sample_training 
			
			incoming.gen = gen = Storage()


			self.samplingLearningMultiple(inp, gen) 

			
			incoming.result.word_loss = self.lossPG(generated_sentence     = gen.w_o, 
													reference_sentence     = incoming.data.resp[1:], 
													generated_distribution = gen.w_pro,
													sentence_length        = gen.length)

			incoming.result.perplexity = torch.exp(incoming.result.word_loss)

		if incoming.now_epoch <= self.param.args.epoch_teacherForcing:
			teacherForcingForward(incoming)
		else:
			policyGradientForward(incoming)

		""" The comments below are for the last version of LossRL"""

		# def unpadSequence(data, length):
		# 	arr = []
		# 	for i in range(length.size):
		# 		arr.append(data[0:length[i], i])
		# 	return arr

		# # generated_sequence_unpad : [batch_size x sequence_length]
		# generated_sequence_unpad     = unpadSequence(gen.w_o, incoming.data.resp_length-1) 
		# # reference_sequence_unpad : [batch_size x sequence_length]
		# reference_sequence_unpad     = unpadSequence(incoming.data.resp[1:], incoming.data.resp_length-1)
		# # generated_distribution_unpad : [batch_size x sequence_length x vocab_size] (distribution for each word)
		# generated_distribution_unpad = unpadSequence(gen.w_pro, incoming.data.resp_length-1)
		
		
		# incoming.result.word_loss = self.lossPG(generated_sequence = generated_sequence_unpad, 
		# 										reference_sequence = reference_sequence_unpad, 
		# 										generated_distribution = generated_distribution_unpad)
		# incoming.result.perplexity = torch.exp(incoming.result.word_loss)

