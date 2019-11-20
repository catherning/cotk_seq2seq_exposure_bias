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
        
        return total_step_counter


    def train_process(self):
        args = self.param.args
        dm = self.param.volatile.dm

        total_step_counter = 1
        while self.now_epoch < args.epochs:
            self.now_epoch += 1
            self.updateOtherWeights()

            dm.restart('train', args.batch_size)
            self.net.train()
            total_step_counter = self.train(args.batch_per_epoch, total_step_counter)
            cur_sampling_proba = 1. - inverse_sigmoid(args.decay_factor,total_step_counter)

            self.net.eval()
            devloss_detail = self.evaluate("dev",cur_sampling_proba)
            self.devSummary(self.now_batch, devloss_detail)
            logging.info("epoch %d, evaluate dev", self.now_epoch)

            testloss_detail = self.evaluate("test",cur_sampling_proba)
            self.testSummary(self.now_batch, testloss_detail)
            logging.info("epoch %d, evaluate test", self.now_epoch)

            self.save_checkpoint(value=devloss_detail.loss.tolist())
        
        return cur_sampling_proba

        

    def evaluate(self, key, sampling_proba):
        args = self.param.args
        dm = self.param.volatile.dm

        dm.restart(key, args.batch_size, shuffle=False)

        result_arr = []
        while True:
            incoming = self.get_next_batch(dm, key, restart=False)
            if incoming is None:
                break
            incoming.args = Storage()
            incoming.args.sampling_proba = sampling_proba

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


    def test(self, key, sampling_proba):
        args = self.param.args
        dm = self.param.volatile.dm

        metric1 = dm.get_teacher_forcing_metric()
        batch_num, batches = self.get_batches(dm, key)
        logging.info("eval teacher-forcing")
        for incoming in tqdm.tqdm(batches, total=batch_num):
            incoming.args = Storage()
            incoming.args.sampling_proba = sampling_proba
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

    def test_process(self, sampling_proba):
        logging.info("Test Start.")
        self.net.eval()
        self.test("dev", sampling_proba)
        test_res = self.test("test")
        logging.info("Test Finish.")
        return test_res