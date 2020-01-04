import os

import numpy as np
import torch
from cotk._utils import hooks
from cotk._utils.file_utils import get_resource_file_path
from cotk.dataloader import OpenSubtitles
from torch import nn

from utils import Storage

def raml_loss(pred, target, sent_size, training_rewards, loss_fn):
    # TODO: check in code and paper, teacher forcing, loss on the golden target or on targets with lower reward, to make model train
    training_rewards = torch.Tensor(training_rewards)
    sent_loss = torch.zeros(target.size()[1])
    for i in range(target.size()[1]):
        sent_loss[i] = loss_fn(pred[:sent_size[i], i, :],
                               target[:sent_size[i], i])
    result = torch.sum(sent_loss * training_rewards) / \
        torch.sum(training_rewards)
    return result

class IWSLT14(OpenSubtitles):
    """A data-loader for IWSLT14 dataset which is a Machine Learning translation dataset
    Arguments:{ARGUMENTS}
    Refer to :class:`.OpenSubtitles` for attributes and methods.
    References:
    """

    ARGUMENTS = OpenSubtitles.ARGUMENTS

    @hooks.hook_dataloader
    def __init__(self, file_id, min_vocab_times=10,
                 max_sent_length=50, invalid_vocab_times=0,
                 num_samples=10, raml_file="samples_iwslt14.txt", tau=0.4, raml=True):
        self._file_id = file_id
        self._file_path = get_resource_file_path(file_id)
        self._min_vocab_times = min_vocab_times
        self._max_sent_length = max_sent_length
        self._invalid_vocab_times = invalid_vocab_times
        
        # RAML specific
        self.raml_mode = raml
        self.n_samples = num_samples
        self.raml_path = os.path.join(self._file_path, raml_file)
        self.tau = tau
        self.raml_path = os.path.join(self._file_path, raml_file)
        super(IWSLT14, self).__init__(file_id=file_id)
        self.raml_data = self.read_raml_sample_file()

    def read_raml_sample_file(self):

        def line2id(line):
            # self.tokenize(line) is more proper than line.split(), but longer
            # self.convert_tokens_to_ids more proper than self.word2id.get(token), but longer
            return [self.go_id] + [self.word2id[token]
                                   for token in line.split()][:self._max_sent_length] + [self.eos_id]
 
        with open(self.raml_path, encoding='utf-8') as raml_file:
            train_data = []
            sample_num = -1
            for line in raml_file.readlines():
                line = line[:-1]
                if line.startswith('***'):
                    continue
                elif line.endswith('samples'):
                    sample_num = eval(line.split()[0])
                    assert sample_num == 1 or sample_num == self.n_samples
                elif line.startswith('source:'):
                    train_data.append(
                        {'source': line[7:], 'targets_id': []}) 
                        # no need to convert source to ids because it is done in load_data
                else:
                    target_line = line.split('|||')
                    token_line = line2id(target_line[0])
                    train_data[-1]['targets_id'].append([token_line, eval(target_line[1])])
                    if sample_num == 1:
                        for i in range(self.n_samples - 1):
                            train_data[-1]['targets_id'].append(
                                [token_line, eval(target_line[1])])

        return train_data

    def get_raml_batch(self, indexes):
        """{LanguageProcessingBase.GET_BATCH_DOC_WITHOUT_RETURNS}
            same def as get_batch(self, indexes) in SingleTurnDialog
        """
        res = {}
        augmented_batch_size = self.batch_size["train"] * self.n_samples
        source_ids = []
        target_ids = []
        scores = []
        res['post_length'] = post_len = np.zeros((augmented_batch_size), dtype=int)
        res['resp_length'] = resp_len = np.zeros((augmented_batch_size), dtype=int)
        res["rewards_ts"] = rewards = np.zeros((augmented_batch_size))

        for source_batch,index in enumerate(indexes):
            training_pair = self.raml_data[index]
            for sample_id, (target,score) in enumerate(training_pair['targets_id']):
                # XXX: input has now <go> token, compared to before. => more coherent with eval & test 
                post = self.data['train']['post'][index]
                source_ids.append(post)
                target_ids.append(target)
                post_len[source_batch*self.n_samples+sample_id] = len(post)
                resp_len[source_batch*self.n_samples+sample_id] = len(target)
                scores.append(score)

        for i in range(0, augmented_batch_size, self.n_samples):
            tmp = np.array(scores[i:i + self.n_samples])
            tmp = np.exp(tmp / self.tau) / np.sum(np.exp(tmp / self.tau))
            for j in range(0, self.n_samples):
                rewards[i+j] = tmp[j]

        res['post'] = res_post = np.zeros(
            (augmented_batch_size, max(post_len)), dtype=int)
        res['resp'] = res_resp = np.zeros(
            (augmented_batch_size, max(resp_len)), dtype=int)
        for i, value in enumerate(source_ids):
            res_post[i, :len(value)] = value
            res_resp[i, :len(target_ids[i])] = target_ids[i]

        res["post_allvocabs"] = res_post.copy()
        res["resp_allvocabs"] = res_resp.copy()
        res_post[res_post >= self.valid_vocab_len] = self.unk_id
        res_resp[res_resp >= self.valid_vocab_len] = self.unk_id

        return res

    def get_next_batch(self, key, ignore_left_samples=False):
        """"Get next batch. It can be called only after Initializing batches (:func:`restart`).

        Arguments:
            key (str): key name of dataset, must be contained in ``self.key_name``.
            ignore_left_samples (bool): If the number of left samples is not equal to
                ``batch_size``, ignore them. This make sure all batches have same number of samples.
                Default: ``False``

        Returns:
            A dict like :func:`get_batch`, or None if the epoch is end.
        """
        if key not in self.key_name:
            raise ValueError("No set named %s." % key)
        if self.batch_size[key] is None:
            raise RuntimeError(
                "Please run restart before calling this function.")
        batch_id = self.batch_id[key]
        start, end = batch_id * \
            self.batch_size[key], (batch_id + 1) * self.batch_size[key]
        if start >= len(self.index[key]):
            return None
        if ignore_left_samples and end > len(self.index[key]):
            return None
        index = self.index[key][start:end]
        if key == "train" and self.raml_mode:
            res = self.get_raml_batch(index)
        else:
            res = self.get_batch(key, index)
        self.batch_id[key] += 1

        return res
