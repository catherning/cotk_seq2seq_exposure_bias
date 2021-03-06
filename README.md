[![Main Repo](https://img.shields.io/badge/Main_project-cotk-blue.svg?logo=github)](https://github.com/thu-coai/cotk)
[![This Repo](https://img.shields.io/badge/Model_repo-pytorch--seq2seq-blue.svg?logo=github)](https://github.com/thu-coai/seq2seq-pytorch)
[![Coverage Status](https://coveralls.io/repos/github/thu-coai/seq2seq-pytorch/badge.svg?branch=master)](https://coveralls.io/github/thu-coai/seq2seq-pytorch?branch=master)
[![Build Status](https://travis-ci.com/thu-coai/seq2seq-pytorch.svg?branch=master)](https://travis-ci.com/thu-coai/seq2seq-pytorch)

# Seq2Seq (PyTorch)

Seq2seq with attention mechanism is a basic model for single turn dialog. In addition, batch normalization and dropout has been applied. You can also choose beamsearch, greedy, random sample, random sample from top-k when decoding.
Note: Beam search is not available for scheduled sampling.

This repository contains Seq2seq models implemented using the Reward Augmented Maximum Likelihood, Scheduled Sampling and Policy Gradient algorithms.

You can refer to the following paper for details:

#### Basic Seq2seq
Sutskever, I., Vinyals, O., & Le, Q. V. (2014). Sequence to sequence learning with neural networks. In *Advances in neural information processing systems* (pp. 3104-3112).

Bahdanau, D., Cho, K., & Bengio, Y. (2015). Neural machine translation by jointly learning to align and translate. In *International Conference on Learning Representation*.

#### RAML algorithm
M. Norouzi et al., “Reward augmented maximum likelihood for neural structured prediction,” in Advances in Neural Information Processing Systems, 2016, no. Ml, (pp. 1731–1739).

X. Ma, P. Yin, J. Liu, G. Neubig, and E. Hovy, “Softmax Q-Distribution Estimation for Structured Prediction: A Theoretical Interpretation for RAML,” pp. 1–23, 2017.

#### Scheduled Sampling

S. Bengio, O. Vinyals, N. Jaitly, and N. Shazeer, “Scheduled sampling for sequence prediction with recurrent neural networks,” in Advances in Neural Information Processing Systems, 2015, vol. 2015-Janua, pp. 1171–1179.


## Require Packages

* **python3**
* cotk
* pytorch == 1.0.0
* tensorboardX >= 1.4

## Quick Start

* Using ``cotk download thu-coai/seq2seq-pytorch/master`` to download codes.
* Execute ``python run.py`` to train the model.
  * The default dataset is ``OpenSubtitles``. You can use ``--dataset`` to specify other ``dataloader`` class and ``--dataid`` to specify other data path (can be a local path, a url or a resources id). For example: ``--dataset OpenSubtitles --dataid resources://OpenSubtitles``
  * It doesn't use pretrained word vector by default setting. You can use ``--wvclass`` to specify ``wordvector`` class and ``--wvpath`` to specify pretrained word embeddings. For example: ``--wvclass gloves``. For example: ``--dataset Glove --dataid resources://Glove300``
  * If you don't have GPUs, you can add `--cpu` for switching to CPU, but it may cost very long time for either training or test.
* You can view training process by tensorboard, the log is at `./tensorboard`.
  * For example, ``tensorboard --logdir=./tensorboard``. (You have to install tensorboard first.)
* After training, execute  ``python run.py --mode test --restore best`` for test.
  * You can use ``--restore filename`` to specify checkpos files, which are in ``./model``. For example: ``--restore pretrained-opensubtitles`` for loading ``./model/pretrained-opensubtitles.model``
  * ``--restore last`` means last checkpo, ``--restore best`` means best checkpos on dev.
  * ``--restore NAME_last`` means last checkpo with model named NAME. The same as``--restore NAME_best``.
* Find results at ``./output``.

## Arguments

```none
    usage: run.py [-h] [--name NAME] [--restore RESTORE] [--mode MODE] [--lr LR]
                  [--eh_size EH_SIZE] [--dh_size DH_SIZE] [--droprate DROPRATE]
                  [--batchnorm] [--decode_mode {max,sample,gumbel,samplek,beam}]
                  [--top_k TOP_K] [--length_penalty LENGTH_PENALTY]
                  [--dataset DATASET] [--dataid DATAID] [--epoch EPOCH]
                  [--batch_per_epoch BATCH_PER_EPOCH] [--wvclass WVCLASS]
                  [--wvid WVID] [--out_dir OUT_DIR] [--log_dir LOG_DIR]
                  [--model_dir MODEL_DIR] [--cache_dir CACHE_DIR] [--cpu]
                  [--debug] [--cache] [--seed SEED]

    A seq2seq model with GRU encoder and decoder. Attention, beamsearch, dropout
    and batchnorm is supported.

    optional arguments:
      -h, --help            show this help message and exit
      --name NAME           The name of your model, used for tensorboard, etc.
                            Default: runXXXXXX_XXXXXX (initialized by current
                            time)
      --model {basic, raml, schedule-sampling, policy-gradien}
                            The type of algorithm. Default: 'basic'
      --restore RESTORE     Checkpos name to load. "NAME_last" for the last
                            checkpo of model named NAME. "NAME_best" means the
                            best checkpo. You can also use "last" and "best",
                            by default use last model you run. Attention:
                            "NAME_last" and "NAME_best" are not guaranteed to work
                            when 2 models with same name run in the same time.
                            "last" and "best" are not guaranteed to work when 2
                            models run in the same time. Default: None (don't load
                            anything)
      --mode MODE           "train" or "test". Default: train
      --lr LR               Learning rate. Default: 0.001
      --eh_size EH_SIZE     Size of encoder GRU
      --dh_size DH_SIZE     Size of decoder GRU
      --droprate DROPRATE   The probability to be zerod in dropout. 0 indicates
                            for don't use dropout
      --batchnorm           Use bathnorm
      --decode_mode {max,sample,gumbel,samplek,beam}
                            The decode strategy when freerun. Choices: max,
                            sample, gumbel(=sample), samplek(sample from topk),
                            beam(beamsearch). Default: beam
      --top_k TOP_K         The top_k when decode_mode == "beam" or "samplek"
      --length_penalty LENGTH_PENALTY
                            The beamsearch penalty for short sentences. The
                            penalty will get larger when this becomes smaller.
      --dataset DATASET     Dataloader class. Default: OpenSubtitles
      --dataid DATAID       Resource id for data set. It can be a resource name or
                            a local path. Default: resources://OpenSubtitles
      --epoch EPOCH         Epoch for training. Default: 100
      --batch_per_epoch BATCH_PER_EPOCH
                            Batches per epoch. Default: 1500
      --wvclass WVCLASS     Wordvector class, none for not using pretrained
                            wordvec. Default: Glove
      --wvid WVID           Resource id for pretrained wordvector. Default:
                            resources://Glove300d
      --out_dir OUT_DIR     Output directory for test output. Default: ./output
      --log_dir LOG_DIR     Log directory for tensorboard. Default: ./tensorboard
      --model_dir MODEL_DIR
                            Checkpos directory for model. Default: ./model
      --cache_dir CACHE_DIR
                            Checkpos directory for cache. Default: ./cache
      --cpu                 Use cpu.
      --debug               Enter debug mode (using ptvsd).
      --cache               Use cache for speeding up load data and wordvec. (It
                            may cause problems when you switch dataset.)
      --seed SEED           Specify random seed. Default: 0

      # RAML parameters
      --raml_file RAML_FILE          
                            The samples and rewards described in RAML. Default:
      --n_samples N_SAMPLES          
                            Number of samples for every target sentence. Default: 10
      --tau TAU             The temperature in RAML algorithm. Default: 0.4

      # Scheduled sampling parameters
      --decay_factor DECAY_FACTOR       
                            The hyperparameter controling the speed of increasing '
                            the probability of sampling from model. Default: 500.

      # Policy Gradient parameters
      --epoch_teacherForcing EPOCH_TF
                            How long to run teacherForcing before running policy gradient. Default: 10
      --nb_sample_training NB_SAMPLE_TRAINING
                            How many samples we take for each batch during policy gradient. Default: 20
      --policy_gradient_reward_mode PG_REWARD_MODE
                            How the policy gradient is applied. Default: mean')
```

## Preliminary Experiments

Based on the best parameters for the basic Seq2seq model
- Encoder & decoder size: 175
- No batchnorm
- Learning rate 0.0005
- Droprate 0.2

with a dev perplexity of 88.688 and a test perplexity of 93.421, we run the other Seq2seq models with the same parameters to compare the performance.

We did the following experiments on `IWSLT14`.
We train during 35 epochs, unless precised.

| Model               | Decode mode     | Dev perplexity  |  Test perplexity  |
| :----------------:  | :------------:  | :-------------: | :--------------:  |
| Basic               | Beam            | 16.266          | 17.302            |
| Basic               | Samplek         | 70.627 | 69.370 |
| RAML                | Beam            | 32.344 | 32.733 |
| RAML                | Samplek         | 75.350 | 74.276 |
| Scheduled Sampling  | Samplek (9 epochs)        | 164.253 | 164.040 |
| Scheduled Sampling  | Max (10 epochs) | 153.723 | 154.406 |
| Policy Gradient     | Beam            | 40.552 | 40.670 |
| Policy Gradient     | Samplek         | 40.552 | 40.670 |
