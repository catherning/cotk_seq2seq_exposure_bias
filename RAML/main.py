# coding:utf-8
import os
import logging
import json

from cotk.dataloader import SingleTurnDialog
from cotk.wordvector import WordVector, Glove

from utils import debug, try_cache, cuda_init, Storage
from raml_helper import IWSLT14
from seq2seq_raml import Seq2seqRAML


def main(args, load_exclude_set, restoreCallback):
    logging.basicConfig(
        filename=0,
        level=logging.DEBUG,
        format='%(asctime)s %(filename)s[line:%(lineno)d] %(message)s',
        datefmt='%H:%M:%S')

    if args.debug:
        debug()
    logging.info(json.dumps(args, indent=2))

    cuda_init(0, args.cuda)

    volatile = Storage()
    volatile.load_exclude_set = load_exclude_set
    volatile.restoreCallback = restoreCallback

    data_class = SingleTurnDialog.load_class(args.dataset)
    data_arg = Storage()
    data_arg.file_id = args.datapath

    # RAML parameters
    data_arg.num_samples = 10 or args.n_samples
    data_arg.raml_file = "samples_iwslt14.txt"
    data_arg.tau = 0.4
    wordvec_class = WordVector.load_class(args.wvclass)

    # XXX: No pretrained vectors. For machine translation with german, wouldn't work ? First try with, if doesn't work, then without
    # would need to init manually the embed layer in network
    # if wordvec_class is None:
    #     wordvec_class = Glove

    def load_dataset(data_arg, wvpath, embedding_size):
        wv = wordvec_class(wvpath)
        dm = data_class(**data_arg)
        return dm, wv.load_matrix(embedding_size, dm.vocab_list)

    if args.cache:
        dm, volatile.wordvec = try_cache(load_dataset, (data_arg, args.wvpath, args.embedding_size),
                                         args.cache_dir, data_class.__name__ + "_" + wordvec_class.__name__)
    else:
        dm, volatile.wordvec = load_dataset(data_arg, args.wvpath, args.embedding_size)

    volatile.dm = dm

    param = Storage()
    param.args = args
    param.volatile = volatile

    model = Seq2seqRAML(param)
    if args.mode == "train":
        model.train_process()
    elif args.mode == "test":
        test_res = model.test_process()

        json.dump(test_res, open("./result.json", "w"))
    else:
        raise ValueError("Unknown mode")
