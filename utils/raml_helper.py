import torch
import torch.nn.functional as F
from torch import nn

# TODO: convert to torch, use args instead of flags

def read_raml_sample_file():
    raml_file = open(FLAGS.raml_file, encoding='utf-8')

    train_data = []
    sample_num = -1
    for line in raml_file.readlines():
        line = line[:-1]
        if line.startswith('***'):
            continue
        elif line.endswith('samples'):
            sample_num = eval(line.split()[0])
            assert sample_num == 1 or sample_num == FLAGS.n_samples
        elif line.startswith('source:'):
            train_data.append({'source': line[7:], 'targets': []})
        else:
            train_data[-1]['targets'].append(line.split('|||'))
            if sample_num == 1:
                for i in range(FLAGS.n_samples - 1):
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


# XXX: should be useless. They were called by cross entropy def of texar
# # From utils.utils
# def sequence_mask(lengths: Union[torch.LongTensor, List[int]],
#                   max_len: Optional[int] = None,
#                   dtype: Optional[torch.dtype] = None,
#                   device: Optional[torch.device] = None):# -> torch.ByteTensor:
#     r"""Return a mask tensor representing the first N positions of each cell.
#     If ``lengths`` has shape ``[d_1, d_2, ..., d_n]`` the resulting tensor
#     ``mask`` has dtype ``dtype`` and shape ``[d_1, d_2, ..., d_n, maxlen]``,
#     with
#     ```
#     mask[i_1, i_2, ..., i_n, j] = (j < lengths[i_1, i_2, ..., i_n])
#     ```
#     Examples:
#     ```python
#     sequence_mask([1, 3, 2], 5)  # [[True, False, False, False, False],
#                                  #  [True,  True,  True, False, False],
#                                  #  [True,  True, False, False, False]]
#     sequence_mask([[1, 3],[2,0]])  # [[[ True, False, False],
#                                    #   [ True,  True,  True]],
#                                    #  [[ True,  True, False],
#                                    #   [False, False, False]]]
#     ```
#     Args:
#         lengths: integer tensor or list of int, all its values <= max_len.
#         max_len: scalar integer tensor, size of last dimension of returned
#             tensor. Default is the maximum value in ``lengths``.
#         dtype: the desired data type of returned tensor. Default: if None,
#             returns :torch:`ByteTensor`.
#         device: the desired device of returned tensor. Default: if None, uses
#             the current device for the default tensor type.
#     Returns:
#         A mask tensor of shape :python:`lengths.shape + (max_len,)`, cast to
#         specified dtype.
#     Raises:
#         ValueError: if ``max_len`` is not a scalar.
#     """
#     if not isinstance(lengths, torch.Tensor):
#         lengths = torch.tensor(lengths, device=device)
#     elif device is None:
#         device = lengths.device
#     lengths: torch.LongTensor
#     if max_len is None:
#         max_len = torch.max(lengths).item()

#     size = lengths.size()
#     row_vector = torch.arange(max_len, device=device, dtype=lengths.dtype).view(
#         *([1] * len(size)), -1).expand(*size, max_len)
#     row_vector = row_vector
#     mask = (row_vector < lengths.unsqueeze(-1)).to(device=device)
#     if dtype is not None:
#         mask = mask.to(dtype=dtype)

#     return mask

# # from utils.shapes
# def transpose_batch_time(inputs: torch.Tensor):# -> torch.Tensor:
#     r"""Transposes inputs between time-major and batch-major.
#     Args:
#         inputs: A Tensor of shape ``[batch_size, max_time, ...]`` (batch-major)
#             or ``[max_time, batch_size, ...]`` (time-major), or a (possibly
#             nested) tuple of such elements.
#     Returns:
#         A (possibly nested tuple of) Tensor with transposed batch and
#         time dimensions of inputs.
#     """
#     return inputs.transpose(0, 1)

# # from utils.shapes
# def mask_sequences(sequence: Union[torch.Tensor, List[int]],
#                    sequence_length: Union[torch.LongTensor, List[int]],
#                    dtype: Optional[torch.dtype] = None,
#                    time_major: bool = False):# -> torch.Tensor:
#     r"""Masks out sequence entries that are beyond the respective sequence
#     lengths. Masks along the time dimension.
#     :attr:`sequence` and :attr:`sequence_length` can either be python
#     arrays or Tensors, respectively. If both are Python arrays (or None), the
#     return will be a Python array as well.
#     Args:
#         sequence: A Tensor or Python array of sequence values.
#             If ``time_major==False`` (default), this must be a Tensor of shape
#             ``[batch_size, max_time, ...]``. The batch and time dimension is
#             exchanged if ``time_major==True``.
#         sequence_length: A Tensor or python array of shape ``[batch_size]``.
#             Time steps beyond the respective sequence lengths will be
#             made zero.
#         dtype (dtype): Type of :attr:`sequence`. If `None`, infer from
#             :attr:`sequence` automatically.
#         time_major (bool): The shape format of the inputs. If `True`,
#             :attr:`sequence` must have shape
#             ``[max_time, batch_size, ...]``.
#             If `False` (default), :attr:`sequence` must have
#             shape ``[batch_size, max_time, ...]``.
#     Returns:
#         The masked sequence, i.e., a Tensor or python array of the same shape
#         as :attr:`sequence` but with masked-out entries (set to zero).
#         If both :attr:`sequence` and :attr:`sequence_length` are python
#         arrays, the returned value is a python array as well.
#     """
#     if not torch.is_tensor(sequence):
#         sequence = torch.tensor(sequence, dtype=dtype)
#     sequence: torch.Tensor

#     rank = sequence.dim()
#     if rank < 2:
#         raise ValueError("`sequence` must be 2D or higher order.")

#     if time_major:
#         sequence = transpose_batch_time(sequence)
#     max_time = sequence.size(1)
#     if dtype is None:
#         dtype = sequence.dtype
#     mask = utils.sequence_mask(sequence_length, max_time, dtype=dtype)
#     mask = mask.view(*mask.size(), *([1] * (rank - 2)))
#     sequence = sequence * mask
#     if time_major:
#         sequence = transpose_batch_time(sequence)

#     return sequence

# # from utils.losses_utils
# def reduce_batch_time(sequence: torch.Tensor,
#                       sequence_length: Optional[torch.LongTensor],
#                       average_across_batch: bool = True,
#                       average_across_timesteps: bool = False,
#                       sum_over_batch: bool = False,
#                       sum_over_timesteps: bool = True):# -> torch.Tensor:
#     r"""Average or sum over the respective dimensions of :attr:`sequence`, which
#     is of shape `[batch_size, max_time, ...]`.
#     Assumes :attr:`sequence` has been properly masked according to
#     :attr:`sequence_length`.
#     Args:
#         sequence: A tensor to reduce.
#         sequence_length: A tensor of shape `[batch_size]`. Time steps beyond
#             the respective sequence lengths will be made zero. If `None`,
#             no masking is performed.
#         average_across_batch (bool): If set, average the sequence across the
#             batch dimension. Must not set `average_across_batch`'
#             and `sum_over_batch` at the same time.
#         average_across_timesteps (bool): If set, average the sequence across
#             the time dimension. Must not set `average_across_timesteps`
#             and `sum_over_timesteps` at the same time.
#         sum_over_batch (bool): If set, sum the sequence across the
#             batch dimension. Must not set `average_across_batch`
#             and `sum_over_batch` at the same time.
#         sum_over_timesteps (bool): If set, sum the sequence across the
#             time dimension. Must not set `average_across_timesteps`
#             and `sum_over_timesteps` at the same time.
#     Returns:
#         A tensor with dimension reduction.
#     """
#     if average_across_timesteps and sum_over_timesteps:
#         raise ValueError("Only one of `average_across_timesteps` and "
#                          "`sum_over_timesteps` can be set.")
#     if average_across_batch and sum_over_batch:
#         raise ValueError("Only one of `average_across_batch` and "
#                          "`sum_over_batch` can be set.")

#     if sum_over_timesteps:
#         sequence = torch.sum(sequence, dim=1)
#     elif average_across_timesteps:
#         if sequence_length is None:
#             sequence = torch.mean(sequence, dim=1)
#         else:
#             sequence = (torch.sum(sequence, dim=1).float() /
#                         sequence_length.float())

#     if sum_over_batch:
#         sequence = torch.sum(sequence, dim=0)
#     elif average_across_batch:
#         sequence = torch.mean(sequence, dim=0)

#     return sequence

# # from utils.losses_utils
# def mask_and_reduce(sequence: torch.Tensor,
#                     sequence_length: Optional[torch.LongTensor],
#                     rank: int = 2,
#                     average_across_batch: bool = True,
#                     average_across_timesteps: bool = False,
#                     average_across_remaining: bool = False,
#                     sum_over_batch: bool = False,
#                     sum_over_timesteps: bool = True,
#                     sum_over_remaining: bool = True,
#                     dtype: Optional[torch.dtype] = None,
#                     time_major: bool = False): # -> torch.Tensor:
#     r"""Masks out sequence entries that are beyond the respective sequence
#     lengths, and reduces (average or sum) away dimensions.
#     This is a combination of :func:`~texar.torch.utils.shapes.mask_sequences`
#     and :func:`~texar.torch.losses.losses_utils.reduce_batch_time`.
#     Args:
#         sequence: A tensor of sequence values.
#             If `time_major=False` (default), this must be a tensor of shape
#             `[batch_size, max_time, d_2, ..., d_rank]`, where the rank of
#             the tensor is specified with :attr:`rank`.
#             The batch and time dimensions are exchanged if `time_major` is True.
#         sequence_length: A tensor of shape `[batch_size]`. Time steps beyond
#             the respective sequence lengths will be made zero. If `None`,
#             no masking is performed.
#         rank (int): The rank of :attr:`sequence`. Must be >= 2. Default is 2,
#             i.e., `sequence` is a 2D Tensor consisting of batch and time
#             dimensions.
#         average_across_timesteps (bool): If set, average the sequence across
#             the time dimension. Must not set `average_across_timesteps`
#             and `sum_over_timesteps` at the same time.
#         average_across_batch (bool): If set, average the sequence across the
#             batch dimension. Must not set `average_across_batch`'
#             and `sum_over_batch` at the same time.
#         average_across_remaining (bool): If set, average the sequence across the
#             remaining dimensions. Must not set `average_across_remaining`'
#             and `sum_over_remaining` at the same time.
#         sum_over_timesteps (bool): If set, sum the sequence across the time
#             dimension. Must not set `average_across_timesteps` and
#             `sum_over_timesteps` at the same time.
#         sum_over_batch (bool): If set, sum the sequence across the batch
#             dimension. Must not set `average_across_batch` and `sum_over_batch`
#             at the same time.
#         sum_over_remaining (bool): If set, sum the sequence across the remaining
#             dimension. Must not set `average_across_remaining` and
#             `sum_over_remaining` at the same time.
#         dtype (torch.dtype): The dtype of the returned mask.
#         time_major (bool): The shape format of the inputs. If `True`,
#             :attr:`sequence` must have shape `[max_time, batch_size, ...]`.
#             If `False` (default), `sequence` must have
#             shape `[batch_size, max_time, ...]`.
#     Returns:
#         A tensor containing the masked and reduced sequence.
#     """
#     if rank < 2:
#         raise ValueError('`rank` must be >= 2.')

#     if time_major:
#         sequence = transpose_batch_time(sequence)

#     if sequence_length is not None:
#         sequence = mask_sequences(sequence,
#                                   sequence_length,
#                                   dtype=dtype,
#                                   time_major=False)
#     if rank > 2:
#         if average_across_remaining and sum_over_remaining:
#             raise ValueError("Only one of `average_across_remaining` and "
#                              "`sum_over_remaining` can be set.")
#         if average_across_remaining:
#             for axis in sorted(list(range(2, rank)), reverse=True):
#                 sequence = torch.mean(sequence, dim=axis)
#         elif sum_over_remaining:
#             for axis in sorted(list(range(2, rank)), reverse=True):
#                 sequence = torch.sum(sequence, dim=axis)

#     sequence = reduce_batch_time(sequence,
#                                  sequence_length,
#                                  average_across_batch,
#                                  average_across_timesteps,
#                                  sum_over_batch,
#                                  sum_over_timesteps)

#     reduce_time = average_across_timesteps or sum_over_timesteps
#     reduce_batch = average_across_batch or sum_over_batch
#     if not reduce_time and not reduce_batch and time_major:
#         sequence = transpose_batch_time(sequence)

#     return sequence

# def sequence_sparse_softmax_cross_entropy(
#         labels: torch.Tensor,
#         logits: torch.Tensor,
#         sequence_length: Optional[torch.LongTensor],
#         average_across_batch: bool = True,
#         average_across_timesteps: bool = False,
#         sum_over_batch: bool = False,
#         sum_over_timesteps: bool = True,
#         time_major: bool = False): # -> torch.Tensor:
#     r"""Computes sparse softmax cross entropy for each time step of sequence
#     predictions.
#     Args:
#         labels: Target class indexes. I.e., classes are mutually exclusive
#             (each entry is in exactly one class).
#             - If :attr:`time_major` is `False` (default), this must be
#               a Tensor of shape `[batch_size, max_time]".
#             - If `time_major` is `True`, this must be a Tensor of shape
#               `[max_time, batch_size].`
#         logits: Unscaled log probabilities. This must have the shape of
#             `[max_time, batch_size, num_classes]` or
#             `[batch_size, max_time, num_classes]` according to
#             the value of `time_major`.
#         sequence_length: A Tensor of shape `[batch_size]`. Time steps beyond
#             the respective sequence lengths will have zero losses.
#         average_across_timesteps (bool): If set, average the loss across
#             the time dimension. Must not set `average_across_timesteps`
#             and `sum_over_timesteps` at the same time.
#         average_across_batch (bool): If set, average the loss across the
#             batch dimension. Must not set `average_across_batch`'
#             and `sum_over_batch` at the same time.
#         sum_over_timesteps (bool): If set, sum the loss across the
#             time dimension. Must not set `average_across_timesteps`
#             and `sum_over_timesteps` at the same time.
#         sum_over_batch (bool): If set, sum the loss across the
#             batch dimension. Must not set `average_across_batch`
#             and `sum_over_batch` at the same time.
#         time_major (bool): The shape format of the inputs. If `True`,
#             :attr:`labels` and :attr:`logits` must have shape
#             `[max_time, batch_size, ...]`. If `False`
#             (default), they must have shape `[batch_size, max_time, ...]`.
#     Returns:
#         A Tensor containing the loss, of rank 0, 1, or 2 depending on the
#         arguments :attr:`{average_across}/{sum_over}_{timesteps}/{batch}`.
#         For example:
#         - If :attr:`sum_over_timesteps` and :attr:`average_across_batch`
#           are `True` (default), the return Tensor is of rank 0.
#         - If :attr:`average_across_batch` is `True` and other arguments are
#           `False`, the return Tensor is of shape `[max_time]`.
#     Example:
#         .. code-block:: python
#             embedder = WordEmbedder(vocab_size=data.vocab.size)
#             decoder = BasicRNNDecoder(vocab_size=data.vocab.size)
#             outputs, _, _ = decoder(
#                 decoding_strategy='train_greedy',
#                 inputs=embedder(data_batch['text_ids']),
#                 sequence_length=data_batch['length']-1)
#             loss = sequence_sparse_softmax_cross_entropy(
#                 labels=data_batch['text_ids'][:, 1:],
#                 logits=outputs.logits,
#                 sequence_length=data_batch['length']-1)
#     """
#     logits = F.log_softmax(logits, dim=2)
#     logits = logits.permute(0, 2, 1)
#     losses = F.nll_loss(logits, labels, reduction='none')

#     losses = mask_and_reduce(losses,
#                              sequence_length,
#                              rank=2,
#                              average_across_batch=average_across_batch,
#                              average_across_timesteps=average_across_timesteps,
#                              sum_over_batch=sum_over_batch,
#                              sum_over_timesteps=sum_over_timesteps,
#                              time_major=time_major)
#     return losses

def raml_loss(pred, target, training_rewards):
    mle_loss = tx.losses.sequence_sparse_softmax_cross_entropy(
        labels=batch['target_text_ids'][:, 1:],
        logits=output.logits,
        sequence_length=batch['target_length'] - 1,
        average_across_batch=False)
    return tf.reduce_sum(mle_loss * training_rewards) /\
           tf.reduce_sum(training_rewards)

    mle_loss = nn.CrossEntropyLoss()(pred,target,reduction="none")



# call of raml_loss
    train_op = tx.core.get_train_op(
        raml_loss(batch, training_outputs, rewards),
        hparams=config_model.opt)

# tensorflow normal seq2seq loss
    train_op = tx.core.get_train_op(
        tx.losses.sequence_sparse_softmax_cross_entropy(
            labels=batch['target_text_ids'][:, 1:],
            logits=training_outputs.logits,
            sequence_length=batch['target_length'] - 1),
        hparams=config_model.opt)

# torch seq2seq2 from texar normal seq2seq loss

    mle_loss = tx.losses.sequence_sparse_softmax_cross_entropy(
        labels=batch['target_text_ids'][:, 1:],
        logits=training_outputs.logits,
        sequence_length=batch['target_length'] - 1)

