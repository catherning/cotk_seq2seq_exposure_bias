import torch
import torch.nn.functional as F
from torch import nn

def read_raml_sample_file(args):
    raml_file = open(args.raml_file, encoding='utf-8')

    train_data = []
    sample_num = -1
    for line in raml_file.readlines():
        line = line[:-1]
        if line.startswith('***'):
            continue
        elif line.endswith('samples'):
            sample_num = eval(line.split()[0])
            assert sample_num == 1 or sample_num == args.n_samples
        elif line.startswith('source:'):
            train_data.append({'source': line[7:], 'targets': []})
        else:
            train_data[-1]['targets'].append(line.split('|||'))
            if sample_num == 1:
                for i in range(args.n_samples - 1):
                    train_data[-1]['targets'].append(line.split('|||'))
    return train_data


# XXX: Def on tensorflow (same with texar torch for tx.losses.seq_sparse...)
# def raml_loss(batch, output, training_rewards):
#     mle_loss = tx.losses.sequence_sparse_softmax_cross_entropy(
#         labels=batch['target_text_ids'][:, 1:],
#         logits=output.logits,
#         sequence_length=batch['target_length'] - 1,
#         average_across_batch=False)
#     return tf.reduce_sum(mle_loss * training_rewards) /\
#            tf.reduce_sum(training_rewards)

def raml_loss(pred, target, training_rewards):
    mle_loss = nn.CrossEntropyLoss(reduction="none")(pred,target)
    return torch.sum(mle_loss * training_rewards) / torch.sum(training_rewards)

    # XXX: tx.losses.sequence_sparse_softmax_cross_entropy output is of rank 0,1 or 2. here should be at least 1. 
    