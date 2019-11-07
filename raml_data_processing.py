import cotk
import argparse

# Use existing dataloader & Write your preprocessor
# LanguageGeneration("./path/to/your_data#your_processor")

# Override the base dataloader
# Class MyDataloader(LanguageGeneration)
# Just define some fields, don’t need to build vocab list again

# coding:utf-8
import os
import json
import numpy as np

from cotk._utils import hooks
from cotk._utils.file_utils import get_resource_file_path
from cotk.dataloader import SingleTurnDialog,OpenSubtitles
from utils import debug, Storage, read_raml_sample_file

class IWSLT14(OpenSubtitles):
    '''A dataloader for IWSLT14 dataset which is a Machine Learning translation dataset
    Arguments:{ARGUMENTS}
    Refer to :class:`.SingleTurnDialog` for attributes and methods.
    References:
    '''

    ARGUMENTS = OpenSubtitles.ARGUMENTS

    @hooks.hook_dataloader
    def __init__(self, file_id, min_vocab_times=10,
                 max_sent_length=50, invalid_vocab_times=0,
                 num_samples=10,raml_file="samples_iwslt14.txt",tau=0.4):
        self._file_id = file_id
        self._file_path = get_resource_file_path(file_id)
        self._min_vocab_times = min_vocab_times
        self._max_sent_length = max_sent_length
        self._invalid_vocab_times = invalid_vocab_times
        
        #RAML specific
        self.raml_data = read_raml_sample_file(os.path.join(self._file_path,raml_file))
        self.n_samples = num_samples
        self.tau = tau
        super(IWSLT14, self).__init__()

    def get_raml_batch(self, indexes):
        '''{LanguageProcessingBase.GET_BATCH_DOC_WITHOUT_RETURNS}
            same def as get_batch(self, indexes) in SingleTurnDialog
        '''

        res = {}
        batch_size = len(indexes) # XXX: either that or give args (batch_size and nb inputs) in input...
        source_buffer, target_buffer = [], []

        for index in indexes:
            training_pair = self.raml_data[index]

            for target in training_pair['targets']:
                source_buffer.append(training_pair['source'])
                target_buffer.append(target)

        source_ids = []
        source_length = []
        target_ids = []
        target_length = []
        scores = []

        trunc_len_src = self._max_sent_length
        trunc_len_tgt = self._max_sent_length
        
        # Source sent to id
        # TODO: can refactor (and same vocab now!) ?
        for sentence in source_buffer:
            ids = [self.word2id[token]
                for token in sentence.split()][:trunc_len_src]
            ids = ids + [self.eos_id]

            source_ids.append(ids)
            source_length.append(len(ids))

        # Target sent to id
        for sentence, score_str in target_buffer:
            ids = [self.go_id]
            ids = ids + [self.word2id[token]
                for token in sentence.split()][:trunc_len_tgt]
            ids = ids + [self.eos_id]

            target_ids.append(ids)
            scores.append(eval(score_str))
            target_length.append(len(ids))

        rewards = []
        for i in range(0, batch_size, self.n_samples):
            tmp = np.array(scores[i:i + self.n_samples])
            tmp = np.exp(tmp / self.tau) / np.sum(np.exp(tmp / self.tau))
            for j in range(0, self.n_samples):
                rewards.append(tmp[j])

        # TODO: padding. Could do differently
        for value in source_ids:
            while len(value) < max(source_length):
                value.append(0)
        for value in target_ids:
            while len(value) < max(target_length):
                value.append(0)

        res['post'] = res_post = np.array(source_ids)
        res['resp'] = res_resp = np.array(target_ids)
        res['post_length'] = np.array(source_length)
        res['resp_length'] = np.array(target_length)
        res["rewards_ts"] = np.array(rewards)

        # XXX: useless to def it ? 
        res["post_allvocabs"] = res_post.copy()
        res["resp_allvocabs"] = res_resp.copy()
        res_post[res_post >= self.valid_vocab_len] = self.unk_id
        res_resp[res_resp >= self.valid_vocab_len] = self.unk_id
        
        return res

    def get_next_raml_batch(self, key, ignore_left_samples=False):
        '''Get next batch. It can be called only after Initializing batches (:func:`restart`).

        Arguments:
            key (str): key name of dataset, must be contained in ``self.key_name``.
            ignore_left_samples (bool): If the number of left samples is not equal to
                ``batch_size``, ignore them. This make sure all batches have same number of samples.
                Default: ``False``

        Returns:
            A dict like :func:`get_batch`, or None if the epoch is end.
        '''
        # if key not in self.key_name:
        #     raise ValueError("No set named %s." % key) # for now, raml file only for training
        if self.batch_size[key] is None:
            raise RuntimeError( \
                "Please run restart before calling this function.")
        batch_id = self.batch_id[key]
        start, end = batch_id * \
            self.batch_size[key], (batch_id + 1) * self.batch_size[key]
        if start >= len(self.index[key]):
            return None
        if ignore_left_samples and end > len(self.index[key]):
            return None
        index = self.index[key][start:end]
        # TODO: only that changed. could same get_next_batch func with raml as bool param if works like that ?
        res = self.get_raml_batch(key, index) 
        self.batch_id[key] += 1
        return res



def main(args):
    data_class = SingleTurnDialog.load_class(args.dataset)
    print(data_class)
    data_arg = Storage()
    data_arg.file_id = args.datapath

    def load_dataset(data_arg):
        return data_class(**data_arg)

    return load_dataset(data_arg)


if __name__ == "__main__":

    args = Storage()

    # args.dataset = "IWSLT14"
    args.datapath = "D:/Documents/THU/Cotk/data/iwslt14"

    args.dataset ='OpenSubtitles'

    args.raml_file = os.path.join(args.datapath,"samples_iwslt14.txt")
    args.n_samples = 10
    data = read_raml_sample_file(args)
    
    # args.datapath ="D:/.cotk_cache/9cf4d4fbf4394c0725c4ad16bf60afd4a40e64c8465bde38d038586118a54888_unzip/opensubtitles/"

    dm = main(args)
    dm.vocab_size

    # attributes
    # all_vocab_list, all_vocab_size
    # batch_id {'dev': 0, 'test': 0, 'train': 0}, batchsize same with None values
    # data dict dev, test, train
    # for dev: post : [[2,25,20,...],...], resp : [[2,95600,...],...]
    # datas_size for dev, test, trian
    # eos_id:3, go_id : 1, pad id 0, unk id 1
    # ext_vocab: tok above
    # index : dev: [i for i in range(..)]
    # key names: [train, dev, test]
    # valid vocab len = vocab size < all_vocab_size
    # word2id
    # _max_sent_length
