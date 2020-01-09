# coding:utf-8

def run(*argv):
    import argparse
    import time
    from utils import Storage

    parser = argparse.ArgumentParser(description='A seq2seq model with GRU encoder and decoder. Attention, beamsearch,\
        dropout and batchnorm is supported. It can train using RAML, Scheduled Sampling or Policy Gradient algorithms.')
    args = Storage()

    parser.add_argument('--name', type=str, default=None,
        help='The name of your model, used for tensorboard, etc. Default: runXXXXXX_XXXXXX (initialized by current time)')
    parser.add_argument('--model', type=str, default="basic",choices=["basic","raml","scheduled-sampling","policy-gradient"],
        help='The type of algorithm. Choices: basic, raml, schedule-sampling, policy-gradient. Default: basic Seq2seq')
    parser.add_argument('--restore', type=str, default=None,
        help='Checkpoints name to load. \
            "NAME_last" for the last checkpoint of model named NAME. "NAME_best" means the best checkpoint. \
            You can also use "last" and "best", by default use last model you run. \
            It can also be an url started with "http". \
            Attention: "NAME_last" and "NAME_best" are not guaranteed to work when 2 models with same name run in the same time. \
            "last" and "best" are not guaranteed to work when 2 models run in the same time.\
            Default: None (don\'t load anything)')
    parser.add_argument('--mode', type=str, default="train",
        help='"train" or "test". Default: train')

    parser.add_argument('--lr', type=float, default=1e-3,
        help='Learning rate. Default: 0.001')
    parser.add_argument('--eh_size', type=int, default=200,
        help='Size of encoder GRU')
    parser.add_argument('--dh_size', type=int, default=200,
        help='Size of decoder GRU')
    parser.add_argument('--droprate', type=float, default=0,
        help='The probability to be zerod in dropout. 0 indicates for don\'t use dropout')
    parser.add_argument('--batchnorm', action='store_true',
        help='Use bathnorm')
    parser.add_argument('--decode_mode', type=str, choices=['max', 'sample', 'gumbel', 'samplek', 'beam'], default='beam',
        help='The decode strategy when freerun. Choices: max, sample, gumbel(=sample), \
            samplek(sample from topk), beam(beamsearch). Default: beam')
    parser.add_argument('--top_k', type=int, default=10,
        help='The top_k when decode_mode == "beam" or "samplek"')
    parser.add_argument('--length_penalty', type=float, default=0.7,
        help='The beamsearch penalty for short sentences. The penalty will get larger when this becomes smaller.')

    parser.add_argument('--dataset', type=str, default='OpenSubtitles',
        help='Dataloader class. Default: OpenSubtitles')
    parser.add_argument('--dataid', type=str, default='resources://OpenSubtitles#OpenSubtitles',
        help='Resource id for data set. It can be a resource name or a local path. Default: resources://OpenSubtitles#OpenSubtitles')
    parser.add_argument('--epoch', type=int, default=100,
        help="Epoch for training. Default: 100")
    parser.add_argument('--batch_per_epoch', type=int, default=1500,
        help="Batches per epoch. Default: 1500")
    parser.add_argument('--wvclass', type=str, default='Glove',
        help="Wordvector class, none for not using pretrained wordvec. Default: Glove")
    parser.add_argument('--wvid', type=str, default="resources://Glove300d",
        help="Resource id for pretrained wordvector. Default: resources://Glove300d")

    parser.add_argument('--out_dir', type=str, default="./output",
        help='Output directory for test output. Default: ./output')
    parser.add_argument('--log_dir', type=str, default="./tensorboard",
        help='Log directory for tensorboard. Default: ./tensorboard')
    parser.add_argument('--model_dir', type=str, default="./model",
        help='Checkpoints directory for model. Default: ./model')
    parser.add_argument('--cache_dir', type=str, default="./cache",
        help='Checkpoints directory for cache. Default: ./cache')
    parser.add_argument('--cpu', action="store_true",
        help='Use cpu.')
    parser.add_argument('--device', type=int, default=0,
        help='Use cpu.')
    parser.add_argument('--debug', action='store_true',
        help='Enter debug mode (using ptvsd).')
    parser.add_argument('--cache', action='store_true',
        help='Use cache for speeding up load data and wordvec. (It may cause problems when you switch dataset.)')
    parser.add_argument('--seed', type=int, default=0,
        help='Specify random seed. Default: 0')

    # RAML parameters
    parser.add_argument('--raml_file', type=str, default='samples_iwslt14.txt',
                        help='the samples and rewards described in RAML')
    parser.add_argument('--n_samples', type=int, default=10,
                        help='number of samples for every target sentence')
    parser.add_argument('--tau', type=float, default=0.4,
                        help='the temperature in RAML algorithm')

    # Scheduled sampling parameters
    parser.add_argument('--decay_factor', type=float, default=500.,
        help='The hyperparameter controling the speed of increasing '
                   'the probability of sampling from model. Default: 500.')

    # Policy Gradient parameters
    parser.add_argument('--epoch_teacherForcing', type=int, default=10,
        help='How long to run teacherForcing before running policy gradient. Default: 10')
    parser.add_argument('--nb_sample_training', type=int, default=20,
        help='How many samples we take for each batch during policy gradient. Default: 20')
    parser.add_argument('--policy_gradient_reward_mode', type=str, default='mean',
        help='How the policy gradient is applied. Default: mean')

    cargs = parser.parse_args(argv)


    # Editing following arguments to bypass command line.
    args.name = cargs.name or time.strftime("run%Y%m%d_%H%M%S", time.localtime())
    args.model = cargs.model
    args.restore = cargs.restore
    args.mode = cargs.mode
    args.dataset = cargs.dataset
    args.datapath = cargs.dataid
    args.epochs = cargs.epoch
    args.wvclass = cargs.wvclass
    args.wvpath = cargs.wvid
    args.out_dir = cargs.out_dir
    args.log_dir = cargs.log_dir
    args.model_dir = cargs.model_dir
    args.cache_dir = cargs.cache_dir
    args.debug = cargs.debug
    args.cache = cargs.cache
    args.cuda = not cargs.cpu
    args.device = cargs.device

    # RAML parameters
    args.raml_file = cargs.raml_file
    args.n_samples = cargs.n_samples
    args.tau = cargs.tau

    # Scheduled sampling parameters
    args.decay_factor = cargs.decay_factor

    # Policy Gradient parameters
    args.epoch_teacherForcing = cargs.epoch_teacherForcing # How long to run teacherForcing before running policy gradient
    args.nb_sample_training   = cargs.nb_sample_training # How many samples we take for each batch during policy gradient
    args.policy_gradient_reward_mode = cargs.policy_gradient_reward_mode # How many samples we take for each batch during policy gradient

    # The following arguments are not controlled by command line.
    args.restore_optimizer = True
    load_exclude_set = []
    restoreCallback = None

    args.batch_per_epoch = cargs.batch_per_epoch
    args.embedding_size = 300
    args.eh_size = cargs.eh_size
    args.dh_size = cargs.dh_size

    args.decode_mode = cargs.decode_mode
    args.top_k = cargs.top_k
    args.length_penalty = cargs.length_penalty

    args.droprate = cargs.droprate
    args.batchnorm = cargs.batchnorm

    args.lr = cargs.lr
    args.batch_size = 1*args.n_samples if args.model=="raml" else 2 #64
    args.batch_num_per_gradient = 4
    args.grad_clip = 5
    args.show_sample = [0]  # show which batch when evaluating at tensorboard
    args.max_sent_length = 50
    args.checkpoint_steps = 20
    args.checkpoint_max_to_keep = 5

    args.seed = cargs.seed

    import random
    random.seed(cargs.seed)
    import torch
    torch.manual_seed(cargs.seed)
    import numpy as np
    np.random.seed(cargs.seed)

    from main import main

    main(args, load_exclude_set, restoreCallback)

if __name__ == '__main__':
    import sys
    run(*sys.argv[1:])
