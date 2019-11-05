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

from cotk.dataloader import SingleTurnDialog
from utils import debug, Storage

class OpenSubtitles(SingleTurnDialog):
    	'''A dataloader for IWSLT14 dataset which is a Machine Learning translation dataset
	Arguments:{ARGUMENTS}
	Refer to :class:`.SingleTurnDialog` for attributes and methods.
	References:
		
	'''

	ARGUMENTS = SingleTurnDialog.ARGUMENTS
	FILE_ID_DEFAULT = None
	VALID_VOCAB_TIMES_DEFAULT = r'''Default: ``10``.'''
	MAX_SENT_LENGTH = r'''Default: ``50``.'''
	INVALID_VOCAB_TIMES_DEFAULT = r'''Default: ``0`` (No unknown words).'''
	TOKENIZER_DEFAULT = r'''Default: ``nltk``'''
	REMAINS_CAPITAL_DEFAULT = r'''Default: ``False``'''
	@hooks.hook_dataloader
	def __init__(self, file_id="resources://OpenSubtitles", min_vocab_times=10, \
			max_sent_length=50, invalid_vocab_times=0, \
			tokenizer="nltk", remains_capital=False\
			):
		super().__init__(file_id, min_vocab_times, max_sent_length, \
			invalid_vocab_times, tokenizer, remains_capital)

def main(args):
    data_class = SingleTurnDialog.load_class(args.dataset)
    data_arg = Storage()
    data_arg.file_id = args.datapath
    
    def load_dataset(data_arg):
        return data_class(**data_arg)

    return load_dataset(data_arg)

if __name__=="__main__":

    args = Storage()

    args.dataset ="iwslt14"
    args.datapath = "D:\\Documents\\THU\\Cotk\\data\\iwslt14"

    # args.dataset ='OpenSubtitles'
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
