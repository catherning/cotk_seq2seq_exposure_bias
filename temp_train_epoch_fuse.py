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

def train(self, raml_train_data, batch_num):
    args = self.param.args

    dm = raml_train_data
    # dm = self.param.volatile.dm
    # datakey = 'train'

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

# fuse train (from seq2seq cotk) into _train_epoch (from raml in tensorflow)
def _train_epoch(self, raml_train_data, batch_num): # have raml_train_data as global variable like in original code ?
    args = self.param.args

    dm = self.param.volatile.dm
    # datakey = 'train' useless because raml_train_data is already train data
    
    # step = 0

    # By default, the processor reads raw data files, performs tokenization,
    # batching and other pre-processing steps, and results in a TF Dataset
    # whose element is a python `dict` including six fields:
        # - "source_text_ids":
        #     An `int64` Tensor of shape `[batch_size, max_time]`
        #     containing the token indexes of source sequences.
        # - "source_length":
        #     An `int` Tensor of shape `[batch_size]` containing the
        #     length of each source sequence in the batch (including BOS and/or
        #     EOS if added).
        # - "target_text_ids":
        #     An `int64` Tensor as "source_text_ids" but for target sequences.
        # - "target_length":
        #     An `int` Tensor of shape `[batch_size]` as "source_length" but for
        #     target sequences.

    train_data = None
    # train_data only for info of batch_size, max_length, vocab to idx
    # batch is incoming
    # TODO: change all call of train_data for its info
    # same for batch in feed dict (=> need to change def of model itself where batch is called in raml tf...??)
    # 2 vocab because 2 languages! take that into account when fuse into normal seq2seq
    
    # XXX: for raml

    incoming = Storage()
    batch_count = 0
    source_buffer, target_buffer = [], []
    random.shuffle(raml_train_data)
    for training_pair in raml_train_data:
        for target in training_pair['targets']:
            source_buffer.append(training_pair['source'])
            target_buffer.append(target)

        if len(target_buffer) != args.batch_size:
            continue
        elif batch_count==batch_num: # TODO: take into account when batch_num*batch_size > total num samples 
            break
    
        source_ids = []
        source_length = []
        target_ids = []
        target_length = []
        scores = []

        trunc_len_src = args.max_sent_length
        trunc_len_tgt = args.max_sent_length
        
        # Source sent to id
        # TODO: can refactor (and same vocab now!)
        for sentence in source_buffer:
            ids = [train_data.source_vocab.token_to_id_map_py[token]
                for token in sentence.split()][:trunc_len_src]
            ids = ids + [train_data.source_vocab.eos_token_id]

            source_ids.append(ids)
            source_length.append(len(ids))

        # Target sent to id
        for sentence, score_str in target_buffer:
            ids = [train_data.target_vocab.bos_token_id]
            ids = ids + [train_data.target_vocab.token_to_id_map_py[token]
                for token in sentence.split()][:trunc_len_tgt]
            ids = ids + [train_data.target_vocab.eos_token_id]

            target_ids.append(ids)
            scores.append(eval(score_str))
            target_length.append(len(ids))

        rewards = []
        for i in range(0, args.batch_size, args.n_samples):
            tmp = np.array(scores[i:i + args.n_samples])
            tmp = np.exp(tmp / args.tau) / np.sum(np.exp(tmp / args.tau))
            for j in range(0, args.n_samples):
                rewards.append(tmp[j])

        # padding. Could do differently
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
        
        batch_count+=1


        # XXX: to here for raml
        loss = sess.run(train_op, feed_dict=feed_dict)

        # print("step={}, loss={:.4f}".format(step, loss),
        #     file=training_log_file)

        # if step % config_data.observe_steps == 0:
        #     print("step={}, loss={:.4f}".format(step, loss))
        # training_log_file.flush()
        # step += 1

def get_batch(self, key, indexes):
        '''{LanguageProcessingBase.GET_BATCH_DOC_WITHOUT_RETURNS}

    Returns:
        (dict): A dict at least contains:

        * **post_length** (:class:`numpy.ndarray`): A 1-d array, the length of post in each batch.
            Size: ``[batch_size]``
        * **post** (:class:`numpy.ndarray`): A 2-d padded array containing words of id form in posts.
            Only provide valid words. ``unk_id`` will be used if a word is not valid.
            Size: ``[batch_size, max(sent_length)]``
        * **post_allvocabs** (:class:`numpy.ndarray`): A 2-d padded array containing words of id
            form in posts. Provide both valid and invalid vocabs.
            Size: ``[batch_size, max(sent_length)]``
        * **resp_length** (:class:`numpy.ndarray`): A 1-d array, the length of response in each batch.
            Size: ``[batch_size]``
        * **resp** (:class:`numpy.ndarray`): A 2-d padded array containing words of id form
            in responses. Only provide valid vocabs. ``unk_id`` will be used if a word is not valid.
            Size: ``[batch_size, max(sent_length)]``
        * **resp_allvocabs** (:class:`numpy.ndarray`):
            A 2-d padded array containing words of id form in responses.
            Provide both valid and invalid vocabs.
            Size: ``[batch_size, max(sent_length)]``

    Examples:
        >>> # all_vocab_list = ["<pad>", "<unk>", "<go>", "<eos>", "how", "are", "you",
        >>> #	"hello", "i", "am", "fine"]
        >>> # vocab_size = 9
        >>> # vocab_list = ["<pad>", "<unk>", "<go>", "<eos>", "how", "are", "you", "hello", "i"]
        >>> dataloader.get_batch('train', [0, 1])
    '''
    if key not in self.key_name:
        raise ValueError("No set named %s." % key)
    res = {}
    batch_size = len(indexes)
    res["post_length"] = np.array(list(map(lambda i: len(self.data[key]['post'][i]), indexes)))
    res["resp_length"] = np.array(list(map(lambda i: len(self.data[key]['resp'][i]), indexes)))
    res_post = res["post"] = np.zeros((batch_size, np.max(res["post_length"])), dtype=int)
    res_resp = res["resp"] = np.zeros((batch_size, np.max(res["resp_length"])), dtype=int)
    for i, j in enumerate(indexes):
        post = self.data[key]['post'][j]
        resp = self.data[key]['resp'][j]
        res_post[i, :len(post)] = post
        res_resp[i, :len(resp)] = resp

    res["post_allvocabs"] = res_post.copy()
    res["resp_allvocabs"] = res_resp.copy()
    res_post[res_post >= self.valid_vocab_len] = self.unk_id
    res_resp[res_resp >= self.valid_vocab_len] = self.unk_id
    return res