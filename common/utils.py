import tensorflow as tf
from common.config import ACT, PAD_TOKEN_IDX


def get_optimizer(*args, **kwargs):
    return tf.compat.v1.train.AdamOptimizer(*args, **kwargs)


def get_train_op(learning_rate, loss, vars, clip_val=None):
    optimizer = get_optimizer(learning_rate)

    grads = tf.gradients(loss, vars)
    if clip_val is not None:
        grads, _ = tf.clip_by_global_norm(grads, clip_val)
    train_op = optimizer.apply_gradients(zip(grads, vars))

    # grads, vars = zip(*optim.compute_gradients(loss, vars, aggregation_method=2))
    # if clip_val is not None:
    #     grads, _ = tf.clip_by_global_norm(grads, clip_val)
    # train_op = optim.apply_gradients(zip(grads, vars))

    return train_op


def clip_and_log(prob):
    return tf.log(tf.clip_by_value(prob, 1e-20, 1.0))


def make_mask(batch_size, keep_prob, units):
    random_tensor = keep_prob
    # 0. if [keep_prob, 1.0) and 1. if [1.0, 1.0 + keep_prob)
    random_tensor += tf.random_uniform(tf.stack([batch_size, units]))
    return tf.floor(random_tensor) / keep_prob


class VariationalDropoutWrapper(tf.contrib.rnn.RNNCell):
    def __init__(self, cell, batch_size, recurrent_keep_prob, input_keep_prob):
        self._cell = cell
        self._recurrent_keep_prob = recurrent_keep_prob
        self._input_keep_prob = input_keep_prob

        self._recurrent_mask = make_mask(batch_size, recurrent_keep_prob, self._cell.state_size[0])
        self._input_mask = self._recurrent_mask

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def __call__(self, inputs, state, scope=None):
        dropped_inputs = inputs * self._input_mask
        dropped_state = (state[0], state[1] * self._recurrent_mask)
        new_h, new_state = self._cell(dropped_inputs, dropped_state, scope)
        return new_h, new_state


def get_mask(acts, use_mask_after_term=False, len_seq=None):
    cond1 = tf.logical_or(tf.equal(acts, ACT['add']), tf.equal(acts, ACT['replace']))
    mask = tf.cast(cond1, tf.int32)

    if use_mask_after_term:
        mask_after_term = get_mask_after_term(acts, len_seq)
        mask *= mask_after_term

    return mask


def get_mask_after_term(acts, len_seq):
    batch_size = acts.shape[0]
    is_terminated = tf.zeros(batch_size, tf.bool)
    mask_arr = []
    for i in range(len_seq):
        _act = acts[:, i]
        _mask = tf.where(is_terminated, tf.zeros(batch_size, tf.bool), tf.ones(batch_size, tf.bool))
        is_terminated = tf.where(tf.equal(_act, ACT['term']), tf.ones(batch_size, tf.bool), is_terminated)

        mask_arr.append(_mask)
    mask = tf.cast(tf.stack(mask_arr, axis=1), tf.int32)
    return mask


def get_mask_for_pad(xs, acts):
    cond1 = tf.logical_or(tf.equal(acts, ACT['add']), tf.equal(acts, ACT['replace']))
    cond = tf.logical_and(cond1, tf.equal(xs, PAD_TOKEN_IDX))

    mask = 1 - tf.cast(cond, tf.int32)

    return mask
