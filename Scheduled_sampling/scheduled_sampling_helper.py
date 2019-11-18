import math

def inverse_sigmoid(args,i):
    return args.decay_factor / (
            args.decay_factor + math.exp(i / args.decay_factor))


@six.add_metaclass(abc.ABCMeta)
class Helper(object):
    """Interface for implementing different decoding strategies in
    :class:`RNN decoders <texar.tf.modules.RNNDecoderBase>` and
    :class:`Transformer decoder <texar.tf.modules.TransformerDecoder>`.
    Adapted from the `tensorflow.contrib.seq2seq` package.
    """

    @abc.abstractproperty
    def batch_size(self):
        """Batch size of tensor returned by `sample`.
        Returns a scalar int32 tensor.
        """
        raise NotImplementedError("batch_size has not been implemented")

    @abc.abstractproperty
    def sample_ids_shape(self):
        """Shape of tensor returned by `sample`, excluding the batch dimension.
        Returns a `TensorShape`.
        """
        raise NotImplementedError("sample_ids_shape has not been implemented")

    @abc.abstractproperty
    def sample_ids_dtype(self):
        """DType of tensor returned by `sample`.
        Returns a DType.
        """
        raise NotImplementedError("sample_ids_dtype has not been implemented")

    @abc.abstractmethod
    def initialize(self, name=None):
        """Returns `(initial_finished, initial_inputs)`."""
        pass

    @abc.abstractmethod
    def sample(self, time, outputs, state, name=None):
        """Returns `sample_ids`."""
        pass

    @abc.abstractmethod
    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        """Returns `(finished, next_inputs, next_state)`."""
        pass


class TrainingHelper(Helper):
    """A helper for use during training. Performs teacher-forcing decoding.
    Returned sample_ids are the argmax of the RNN output logits.
    Note that for teacher-forcing decoding, Texar's decoders provide a simpler
    interface by specifying `decoding_strategy='train_greedy'` when calling a
    decoder (see, e.g.,,
    :meth:`RNN decoder <texar.tf.modules.RNNDecoderBase._build>`). In this case,
    use of TrainingHelper is not necessary.
    """

    def __init__(self, inputs, sequence_length, time_major=False, name=None):
        """Initializer.
        Args:
          inputs: A (structure of) input tensors.
          sequence_length: An int32 vector tensor.
          time_major: Python bool.  Whether the tensors in `inputs` are time major.
            If `False` (default), they are assumed to be batch major.
          name: Name scope for any created operations.
        Raises:
          ValueError: if `sequence_length` is not a 1D tensor.
        """
        with ops.name_scope(name, "TrainingHelper", [inputs, sequence_length]):
            inputs = ops.convert_to_tensor(inputs, name="inputs")
            self._inputs = inputs
            if not time_major:
                inputs = nest.map_structure(_transpose_batch_time, inputs)

            self._input_tas = nest.map_structure(_unstack_ta, inputs)
            self._sequence_length = ops.convert_to_tensor(
                sequence_length, name="sequence_length")
            if self._sequence_length.get_shape().ndims != 1:
                raise ValueError(
                    "Expected sequence_length to be a vector, but received shape: %s" %
                    self._sequence_length.get_shape())

            self._zero_inputs = nest.map_structure(
                lambda inp: array_ops.zeros_like(inp[0, :]), inputs)
            self._start_inputs = self._zero_inputs
            self._batch_size = shape_list(sequence_length)[0]

    @property
    def inputs(self):
        return self._inputs

    @property
    def sequence_length(self):
        return self._sequence_length

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def sample_ids_shape(self):
        return tensor_shape.TensorShape([])

    @property
    def sample_ids_dtype(self):
        return dtypes.int32

    def initialize(self, name=None):
        with ops.name_scope(name, "TrainingHelperInitialize"):
            finished = math_ops.equal(0, self._sequence_length)
            all_finished = math_ops.reduce_all(finished)
            next_inputs = control_flow_ops.cond(
                all_finished, lambda: self._zero_inputs,
                lambda: nest.map_structure(lambda inp: inp.read(0), self._input_tas))
            return (finished, next_inputs)

    def sample(self, time, outputs, name=None, **unused_kwargs):
        """Gets a sample for one step."""
        with ops.name_scope(name, "TrainingHelperSample", [time, outputs]):
            sample_ids = math_ops.cast(
                math_ops.argmax(outputs, axis=-1), dtypes.int32)
            return sample_ids

    def next_inputs(self, time, outputs, state, name=None, **unused_kwargs):
        """Gets the inputs for next step."""
        with ops.name_scope(name, "TrainingHelperNextInputs",
                            [time, outputs, state]):
            next_time = time + 1
            finished = (next_time >= self._sequence_length)
            all_finished = math_ops.reduce_all(finished)

            def read_from_ta(inp):
                return inp.read(next_time)

            next_inputs = control_flow_ops.cond(
                all_finished, lambda: self._zero_inputs,
                lambda: nest.map_structure(read_from_ta, self._input_tas))
            return (finished, next_inputs, state)


class ScheduledEmbeddingTrainingHelper(TrainingHelper):
    """A training helper that adds scheduled sampling.
    Returns -1s for sample_ids where no sampling took place; valid sample id
    values elsewhere.
    """

    def __init__(self, inputs, sequence_length, embedding, sampling_probability,
                 time_major=False, seed=None, scheduling_seed=None, name=None):
        """Initializer.
        Args:
          inputs: A (structure of) input tensors.
          sequence_length: An int32 vector tensor.
          embedding: A callable or the `params` argument for `embedding_lookup`.
            If a callable, it can take a vector tensor of token `ids`,
            or take two arguments (`ids`, `times`), where `ids` is a vector
            tensor of token ids, and `times` is a vector tensor of current
            time steps (i.e., position ids). The latter case can be used when
            attr:`embedding` is a combination of word embedding and position
            embedding.
          sampling_probability: A 0D `float32` tensor: the probability of sampling
            categorically from the output ids instead of reading directly from the
            inputs.
          time_major: Python bool.  Whether the tensors in `inputs` are time major.
            If `False` (default), they are assumed to be batch major.
          seed: The sampling seed.
          scheduling_seed: The schedule decision rule sampling seed.
          name: Name scope for any created operations.
        Raises:
          ValueError: if `sampling_probability` is not a scalar or vector.
        """
        with ops.name_scope(name, "ScheduledEmbeddingSamplingWrapper",
                            [embedding, sampling_probability]):
            if callable(embedding):
                self._embedding_fn = embedding
            else:
                self._embedding_fn = (
                    lambda ids: embedding_ops.embedding_lookup(embedding, ids))

            self._embedding_args_cnt = len(get_args(self._embedding_fn))
            if self._embedding_args_cnt != 1 and self._embedding_args_cnt != 2:
                raise ValueError('`embedding` should expect 1 or 2 arguments.')

            self._sampling_probability = ops.convert_to_tensor(
                sampling_probability, name="sampling_probability")
            if self._sampling_probability.get_shape().ndims not in (0, 1):
                raise ValueError(
                    "sampling_probability must be either a scalar or a vector. "
                    "saw shape: %s" % (self._sampling_probability.get_shape()))
            self._seed = seed
            self._scheduling_seed = scheduling_seed
            super(ScheduledEmbeddingTrainingHelper, self).__init__(
                inputs=inputs,
                sequence_length=sequence_length,
                time_major=time_major,
                name=name)

    def initialize(self, name=None):
        return super(ScheduledEmbeddingTrainingHelper, self).initialize(
            name=name)

    def sample(self, time, outputs, state, name=None):
        """Gets a sample for one step."""
        with ops.name_scope(name, "ScheduledEmbeddingTrainingHelperSample",
                            [time, outputs, state]):
            # Return -1s where we did not sample, and sample_ids elsewhere
            select_sampler = tfpd.Bernoulli(
                probs=self._sampling_probability, dtype=dtypes.bool)
            select_sample = select_sampler.sample(
                sample_shape=self.batch_size, seed=self._scheduling_seed)
            sample_id_sampler = tfpd.Categorical(logits=outputs)
            return array_ops.where(
                select_sample,
                sample_id_sampler.sample(seed=self._seed),
                gen_array_ops.fill([self.batch_size], -1))

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        """Gets the outputs for next step."""
        with ops.name_scope(name, "ScheduledEmbeddingTrainingHelperNextInputs",
                            [time, outputs, state, sample_ids]):
            (finished, base_next_inputs, state) = (
                super(ScheduledEmbeddingTrainingHelper, self).next_inputs(
                    time=time,
                    outputs=outputs,
                    state=state,
                    sample_ids=sample_ids,
                    name=name))

            def maybe_sample():
                """Perform scheduled sampling."""
                where_sampling = math_ops.cast(
                    array_ops.where(sample_ids > -1), dtypes.int32)
                where_not_sampling = math_ops.cast(
                    array_ops.where(sample_ids <= -1), dtypes.int32)
                sample_ids_sampling = array_ops.gather_nd(sample_ids, where_sampling)
                inputs_not_sampling = array_ops.gather_nd(
                    base_next_inputs, where_not_sampling)

                if self._embedding_args_cnt == 1:
                    sampled_next_inputs = self._embedding_fn(
                        sample_ids_sampling)
                elif self._embedding_args_cnt == 2:
                    # Prepare the position embedding of the next step
                    times = tf.ones(self._batch_size,
                                    dtype=tf.int32) * (time + 1)
                    sampled_next_inputs = self._embedding_fn(
                        sample_ids_sampling, times)
                base_shape = array_ops.shape(base_next_inputs)
                return (array_ops.scatter_nd(indices=where_sampling,
                                             updates=sampled_next_inputs,
                                             shape=base_shape)
                        + array_ops.scatter_nd(indices=where_not_sampling,
                                               updates=inputs_not_sampling,
                                               shape=base_shape))

            all_finished = math_ops.reduce_all(finished)
            next_inputs = control_flow_ops.cond(
                all_finished, lambda: base_next_inputs, maybe_sample)
            return (finished, next_inputs, state)

