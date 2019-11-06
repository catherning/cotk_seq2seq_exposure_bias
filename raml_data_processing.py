import cotk
import argparse

# Use existing dataloader & Write your preprocessor
# LanguageGeneration("./path/to/your_data#your_processor")

# Override the base dataloader
# Class MyDataloader(LanguageGeneration)
# Just define some fields, don’t need to build vocab list again

# coding:utf-8
import os
import sys
import json

from cotk._utils import hooks
from cotk._utils.file_utils import get_resource_file_path
from cotk.dataloader import SingleTurnDialog
from utils import debug, Storage, read_raml_sample_file

class IWSLT14(SingleTurnDialog):
    '''A dataloader for IWSLT14 dataset which is a Machine Learning translation dataset
    Arguments:{ARGUMENTS}
    Refer to :class:`.SingleTurnDialog` for attributes and methods.
    References:
    '''

    ARGUMENTS = SingleTurnDialog.ARGUMENTS

    @hooks.hook_dataloader
    def __init__(self, file_id, min_vocab_times=10,
                 max_sent_length=50, invalid_vocab_times=0):
        self._file_id = file_id
        self._file_path = get_resource_file_path(file_id)
        self._min_vocab_times = min_vocab_times
        self._max_sent_length = max_sent_length
        self._invalid_vocab_times = invalid_vocab_times
        super(IWSLT14, self).__init__()

    def _load_data(self):
        r'''Loading dataset, invoked during the initialization of :class:`SingleTurnDialog`.
        return vocab_list, valid_vocab_len, data, data_size
        '''
        
        origin_data = {}
        for key in self.key_name:
            f_file = open("%s/opensub_pair_%s.post" % (self._file_path, key), 'r', encoding='utf-8')
            g_file = open("%s/opensub_pair_%s.response" % (self._file_path, key), 'r', encoding='utf-8')
            origin_data[key] = {}
            origin_data[key]['post'] = list(map(lambda line: line.split(), f_file.readlines()))
            origin_data[key]['resp'] = list(map(lambda line: line.split(), g_file.readlines()))

        raw_vocab_list = list(chain(*(origin_data['train']['post'] + origin_data['train']['resp'])))
        # Important: Sort the words preventing the index changes between different runs
        vocab = sorted(Counter(raw_vocab_list).most_common(), key=lambda pair: (-pair[1], pair[0]))
        left_vocab = list(filter(lambda x: x[1] >= self._min_vocab_times, vocab))
        vocab_list = self.ext_vocab + list(map(lambda x: x[0], left_vocab))
        valid_vocab_len = len(vocab_list)
        valid_vocab_set = set(vocab_list)

        for key in self.key_name:
            if key == 'train':
                continue
            raw_vocab_list.extend(list(chain(*(origin_data[key]['post'] + origin_data[key]['resp']))))
        vocab = sorted(Counter(raw_vocab_list).most_common(), \
                       key=lambda pair: (-pair[1], pair[0]))
        left_vocab = list( \
            filter( \
                lambda x: x[1] >= self._invalid_vocab_times and x[0] not in valid_vocab_set, \
                vocab))
        vocab_list.extend(list(map(lambda x: x[0], left_vocab)))

        print("valid vocab list length = %d" % valid_vocab_len)
        print("vocab list length = %d" % len(vocab_list))

        word2id = {w: i for i, w in enumerate(vocab_list)}
        line2id = lambda line: ([self.go_id] + \
                    list(map(lambda word: word2id[word] if word in word2id else self.unk_id, line)) + \
                    [self.eos_id])[:self._max_sent_length]

        data = {}
        data_size = {}
        for key in self.key_name:
            data[key] = {}

            data[key]['post'] = list(map(line2id, origin_data[key]['post']))
            data[key]['resp'] = list(map(line2id, origin_data[key]['resp']))
            data_size[key] = len(data[key]['post'])
            vocab = list(chain(*(origin_data[key]['post'] + origin_data[key]['resp'])))
            vocab_num = len(vocab)
            oov_num = len(list(filter(lambda word: word not in word2id, vocab)))
            invalid_num = len( \
                list( \
                    filter( \
                        lambda word: word not in valid_vocab_set, \
                        vocab))) - oov_num
            length = list(map(len, origin_data[key]['post'] + origin_data[key]['resp']))
            cut_num = np.sum(np.maximum(np.array(length) - self._max_sent_length + 1, 0))
            print("%s set. invalid rate: %f, unknown rate: %f, max length before cut: %d, \
                    cut word rate: %f" % \
                    (key, invalid_num / vocab_num, oov_num / vocab_num, max(length), cut_num / vocab_num))

        return vocab_list, valid_vocab_len, data, data_size


    def tokenize(self, sentence):
        pass

    def get_batch(self, key, indexes):
        pass


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
