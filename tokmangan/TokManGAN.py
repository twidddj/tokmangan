import tensorflow as tf
from tensorflow.python.ops import tensor_array_ops, control_flow_ops
import numpy as np

import common.attention_utils as attention_utils
from tokmangan.modules import Discriminator, Critic
from common.config import ACT, START_TOKEN_IDX, END_TOKEN_IDX, PAD_TOKEN_IDX
from common.utils import get_mask, VariationalDropoutWrapper, make_mask, clip_and_log, get_train_op


def transform_seed_with_token(generator, seed, acts, act_dict=ACT):
    batch_size, sequence_length = generator.batch_size, generator.sequence_length

    masked_inputs = tensor_array_ops.TensorArray(dtype=tf.int32, size=sequence_length, dynamic_size=False,
                                                 infer_shape=True, name='masked_inputs')
    ta_acts = tensor_array_ops.TensorArray(dtype=tf.int32, size=sequence_length, name='ta_acts')
    ta_acts = ta_acts.unstack(tf.transpose(acts, perm=[1, 0]))

    input_add = tf.constant(generator.num_vocabulary, dtype=tf.int32, shape=[batch_size, ])
    input_replace = tf.constant(generator.num_vocabulary + 1, dtype=tf.int32, shape=[batch_size, ])
    input_ignore = tf.constant(generator.num_vocabulary + 2, dtype=tf.int32, shape=[batch_size, ])
    input_term = tf.constant(generator.num_vocabulary + 3, dtype=tf.int32, shape=[batch_size, ])

    def _recurrence(i, seed_idx, masked_inputs, ta_acts):
        current_act = ta_acts.read(i)
        # is_add = tf.equal(current_act, ACT['add'])

        is_use = tf.equal(current_act, act_dict['skip'])
        is_ignore = tf.equal(current_act, act_dict['pass'])
        is_replace = tf.equal(current_act, act_dict['replace'])
        is_term = tf.equal(current_act, act_dict['term'])

        is_seed_action = tf.logical_or(is_use, is_ignore)
        is_seed_action = tf.logical_or(is_seed_action, is_replace)

        seed_idx = tf.where(tf.math.greater_equal(seed_idx, generator.seed_sequence_length), generator.maximum_seed_idx, seed_idx)
        coords = tf.concat([tf.reshape(tf.range(batch_size), [-1, 1]), tf.reshape(seed_idx, [-1, 1])], -1)
        sampled_from_seed = tf.gather_nd(seed, coords)

        current_input = tf.where(is_use, sampled_from_seed, input_add)
        current_input = tf.where(is_replace, input_replace, current_input)
        current_input = tf.where(is_ignore, input_ignore, current_input)
        current_input = tf.where(is_term, input_term, current_input)

        seed_idx = tf.where(is_seed_action, seed_idx + 1, seed_idx)
        masked_inputs = masked_inputs.write(i, current_input)

        return i + 1, seed_idx, masked_inputs, ta_acts

    i, seed_idx, masked_inputs, ta_acts = control_flow_ops.while_loop(
        cond=lambda i, _1, _2, _3: i < sequence_length,
        body=_recurrence,
        loop_vars=(
            tf.constant(0, dtype=tf.int32),
            tf.constant(0, shape=[batch_size]),
            masked_inputs,
            ta_acts))

    transformed = masked_inputs.stack()
    transformed = tf.transpose(transformed, perm=[1, 0])
    return transformed


class TokManGAN(object):
    def __init__(self, num_vocabulary, batch_size, emb_dim, hidden_dim, sequence_length, seed_length,
                 reward_gamma=0.9, is_training=False, gen_vd_keep_prob=1, dis_vd_keep_prob=1):

        self.is_training = is_training
        self.num_vocabulary = num_vocabulary
        self.batch_size = batch_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim

        self.sequence_length = sequence_length
        self.seed_sequence_length = seed_length
        self.n_actions = len(ACT)

        self.n_rnn_layers = 2

        self.reward_gamma = reward_gamma
        self.clip_val = 5.0

        self.gen_vd_keep_prob = gen_vd_keep_prob
        self.dis_vd_keep_prob = dis_vd_keep_prob

        self.start_token = tf.constant([START_TOKEN_IDX] * self.batch_size, dtype=tf.int32)
        self.end_token = tf.constant([END_TOKEN_IDX] * self.batch_size, dtype=tf.int32)
        self.pad_token = tf.constant([PAD_TOKEN_IDX] * self.batch_size, dtype=tf.int32)
        self.maximum_seed_idx = tf.constant([self.seed_sequence_length - 1] * batch_size, dtype=tf.int32)

        # placeholder definition
        self.x = tf.placeholder(tf.int32, shape=[self.batch_size, self.sequence_length], name="target_x")
        self.x_len = tf.placeholder(tf.int32, shape=[self.batch_size, ], name="x_len")  # n_x
        self.seed = tf.placeholder(tf.int32, shape=[self.batch_size, self.seed_sequence_length], name="seed")
        self.acts = tf.placeholder(tf.int32, shape=[self.batch_size, self.sequence_length], name="target_acts")
        self.seed_len = tf.placeholder(tf.int32, shape=[self.batch_size, ], name="seed_len")  # n_seed
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        with tf.variable_scope('TokGANGenerator'):
            self.create()
            self.create_network()
            if self.is_training:
                self.create_pretrain_network()

        self.g_params = [var for var in tf.trainable_variables() if var.name.startswith('TokGANGenerator')]

        if self.is_training:
            real_transformed_xs = transform_seed_with_token(self, self.seed, self.acts)
            fake_transformed_xs = transform_seed_with_token(self, self.seed, self.gen_act)

            self.discriminator = Discriminator(self)
            real_predictions = self.discriminator.create_network(self.x, real_transformed_xs, reuse=False)
            real_missing = tf.cast(get_mask(self.acts), tf.float32)
            fake_predictions = self.discriminator.create_network(self.gen_x, fake_transformed_xs, reuse=True)
            fake_missing = tf.cast(get_mask(self.gen_act), tf.float32)
            self.dis_loss_fake, self.dis_loss_real, self.dis_updates = \
                self.discriminator.create_loss(fake_predictions, real_predictions,
                                               self.gen_x, self.x,
                                               fake_missing, real_missing)

            self.critic = Critic(self, self.g_embeddings, self.gen_x, self.gen_act)

            self.pretrain_loss, self.pretrain_act_loss, self.pretrain_updates = self.create_pretrain_loss()
            self.g_loss, self.g_updates = self.create_adversarial_loss(fake_predictions)
            self.dis_updates = tf.group(self.dis_updates, self.critic_updates)


    def create(self, reuse=None):
        init_embeddings = tf.random_uniform([self.num_vocabulary, self.emb_dim], -1.0, 1.0)
        self.g_embeddings = tf.get_variable("embeddings", initializer=init_embeddings)

        def lstm_cell():
            return tf.contrib.rnn.BasicLSTMCell(self.hidden_dim, forget_bias=0.0, state_is_tuple=True, reuse=reuse)

        attn_cell = lstm_cell
        if self.is_training and self.gen_vd_keep_prob < 1:
            def attn_cell():
                return VariationalDropoutWrapper(lstm_cell(), self.batch_size, self.gen_vd_keep_prob,
                                                 self.gen_vd_keep_prob)

        with tf.variable_scope('seed_encoder') as scope:
            embedded_seed = tf.nn.embedding_lookup(self.g_embeddings, self.seed)

            cell = tf.contrib.rnn.MultiRNNCell([attn_cell() for _ in range(self.n_rnn_layers)], state_is_tuple=True)
            state = cell.zero_state(self.batch_size, tf.float32)

            outputs, state = tf.nn.dynamic_rnn(cell, embedded_seed, initial_state=state, scope=scope)

            def _make_mask(batch_size, keep_prob, units):
                random_tensor = keep_prob
                random_tensor += tf.random_uniform(
                    tf.stack([batch_size, 1, units]))
                return tf.floor(random_tensor) / keep_prob

            if self.is_training:
                output_mask = _make_mask(self.batch_size, self.gen_vd_keep_prob, self.hidden_dim)
                outputs *= output_mask

            self.encoded_seed = outputs
            # Initial states
            # self.h0 = self.g_recurrent_unit.zero_state(self.batch_size, tf.float32)
            self.h0 = state

        with tf.variable_scope('attention'):
            print("encoded seed:", self.encoded_seed.shape)
            (self.attention_keys, self.attention_values, _,
             self.attention_construct_fn) = attention_utils.prepare_attention(
                self.encoded_seed, 'luong', num_units=self.hidden_dim)

        with tf.variable_scope('generator'):
            def lstm_cell():
                return tf.contrib.rnn.BasicLSTMCell(self.hidden_dim, forget_bias=0.0, state_is_tuple=True, reuse=reuse)

            attn_cell = lstm_cell
            if self.gen_vd_keep_prob < 1:
                def attn_cell():
                    return VariationalDropoutWrapper(lstm_cell(), self.batch_size, self.gen_vd_keep_prob, self.gen_vd_keep_prob)

            self.g_recurrent_unit = tf.contrib.rnn.MultiRNNCell([attn_cell() for _ in range(self.n_rnn_layers)],
                                                                state_is_tuple=True)
            self.g_output_unit = self.create_output_unit()  # maps h_t to o_t (output token logits)
            self.g_act_unit = self.create_action_unit()

    def create_network(self):
        gen_log_p = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_length, dynamic_size=False,
                                                 infer_shape=True)
        gen_x = tensor_array_ops.TensorArray(dtype=tf.int32, size=self.sequence_length, dynamic_size=False,
                                             infer_shape=True)
        gen_act_log_p = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_length, dynamic_size=False,
                                                     infer_shape=True)
        gen_mask_act_p = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_length, dynamic_size=False,
                                                      infer_shape=True)
        gen_act = tensor_array_ops.TensorArray(dtype=tf.int32, size=self.sequence_length, dynamic_size=False,
                                               infer_shape=True)

        output_mask = None
        if self.gen_vd_keep_prob < 1:
            output_mask = make_mask(self.batch_size, self.gen_vd_keep_prob, self.hidden_dim)

        def _g_recurrence(i, x_t, prev_h_state, gen_log_p, gen_x, gen_act_log_p, gen_act, gen_mask_act_p, seed_idx):
            out, state = self.g_recurrent_unit(x_t, prev_h_state)  # hidden_memory_tuple
            out = self.attention_construct_fn(out, self.attention_keys, self.attention_values)

            if output_mask is not None:
                out *= output_mask

            o_t = self.g_output_unit(out)  # batch x vocab , logits not prob
            a_t = self.g_act_unit(out)  # batch x len(ACT)

            prob = tf.nn.softmax(o_t)
            act_prob = tf.nn.softmax(a_t)
            log_prob = clip_and_log(prob)
            log_act_prob = clip_and_log(act_prob)

            mask_act_p = act_prob[:, ACT['add']] + act_prob[:, ACT['replace']]
            gen_mask_act_p = gen_mask_act_p.write(i, mask_act_p)

            next_act = tf.squeeze(tf.random.categorical(log_act_prob, 1))
            next_act = tf.cast(next_act, tf.int32)

            one_hot_next_act = tf.one_hot(next_act, self.n_actions, 1.0, 0.0)
            log_p_next_act = tf.multiply(one_hot_next_act, log_act_prob)
            gen_act_log_p = gen_act_log_p.write(i, tf.reduce_sum(log_p_next_act, 1))

            cond = tf.logical_or(tf.equal(next_act, ACT['skip']), tf.equal(next_act, ACT['pass']))
            cond = tf.logical_or(cond, tf.equal(next_act, ACT['replace']))

            next_token = tf.squeeze(tf.random.categorical(log_prob, 1))
            next_token = tf.cast(next_token, tf.int32)

            seed_idx = tf.where(tf.math.greater_equal(seed_idx, self.seed_sequence_length), self.maximum_seed_idx,
                                seed_idx)
            coords = tf.concat([tf.reshape(tf.range(self.batch_size), [-1, 1]), tf.reshape(seed_idx, [-1, 1])], -1)
            sampled_from_seed = tf.gather_nd(self.seed, coords)

            next_token = tf.where(tf.equal(next_act, ACT['skip']), sampled_from_seed, next_token)
            next_token = tf.where(tf.equal(next_act, ACT['term']), self.end_token, next_token)
            next_token = tf.where(tf.equal(next_act, ACT['pass']), self.pad_token, next_token)

            seed_idx = tf.where(cond, seed_idx + 1, seed_idx)

            x_tp1 = tf.nn.embedding_lookup(self.g_embeddings, next_token)  # batch x emb_dim

            one_hot_next_token = tf.one_hot(next_token, self.num_vocabulary, 1.0, 0.0)
            log_p_next_token = tf.multiply(one_hot_next_token, log_prob)

            gen_log_p = gen_log_p.write(i, tf.reduce_sum(log_p_next_token, 1))  # [batch_size] , prob
            gen_x = gen_x.write(i, next_token)  # indices, batch_size
            gen_act = gen_act.write(i, next_act)
            return i + 1, x_tp1, state, gen_log_p, gen_x, gen_act_log_p, gen_act, gen_mask_act_p, seed_idx

        _, _, _, self.gen_log_p, self.gen_x, self.gen_act_log_p, self.gen_act, self.gen_mask_act_p, _ = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3, _4, _5, _6, _7, _8: i < self.sequence_length,
            body=_g_recurrence,
            loop_vars=(tf.constant(0, dtype=tf.int32),
                       tf.nn.embedding_lookup(self.g_embeddings, self.start_token),
                       self.h0, gen_log_p, gen_x, gen_act_log_p, gen_act, gen_mask_act_p,
                       tf.constant(0, shape=[self.batch_size])))

        self.gen_x = self.gen_x.stack()  # seq_length x batch_size
        self.gen_x = tf.transpose(self.gen_x, perm=[1, 0])  # batch_size x seq_length
        self.gen_x = tf.reshape(self.gen_x, shape=[self.batch_size, self.sequence_length])

        self.gen_act = self.gen_act.stack()  # seq_length x batch_size
        self.gen_act = tf.transpose(self.gen_act, perm=[1, 0])  # batch_size x seq_length
        self.gen_act = tf.reshape(self.gen_act, shape=[self.batch_size, self.sequence_length])

        self.gen_log_p = self.gen_log_p.stack()  # seq_length x batch_size
        self.gen_log_p = tf.transpose(self.gen_log_p, perm=[1, 0])  # batch_size x seq_length
        self.gen_log_p = tf.reshape(self.gen_log_p, shape=[self.batch_size, self.sequence_length])

        self.gen_act_log_p = self.gen_act_log_p.stack()  # seq_length x batch_size
        self.gen_act_log_p = tf.transpose(self.gen_act_log_p, perm=[1, 0])  # batch_size x seq_length
        self.gen_act_log_p = tf.reshape(self.gen_act_log_p, shape=[self.batch_size, self.sequence_length])

        self.gen_mask_act_p = self.gen_mask_act_p.stack()  # seq_length x batch_size
        self.gen_mask_act_p = tf.transpose(self.gen_mask_act_p, perm=[1, 0])  # batch_size x seq_length
        self.gen_mask_act_p = tf.reshape(self.gen_mask_act_p, shape=[self.batch_size, self.sequence_length])

    def create_pretrain_network(self):
        with tf.variable_scope('attention'):
            (self.attention_keys, self.attention_values, _,
             self.attention_construct_fn) = attention_utils.prepare_attention(
                self.encoded_seed, 'luong', num_units=self.hidden_dim, reuse=True)

        # supervised pretraining for generator
        g_predictions = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_length, dynamic_size=False,
                                                     infer_shape=True)
        a_predictions = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_length, dynamic_size=False,
                                                     infer_shape=True)

        # processed for batch
        with tf.device("/cpu:0"):
            self.processed_x = tf.transpose(tf.nn.embedding_lookup(self.g_embeddings, self.x),
                                            perm=[1, 0, 2])  # seq_length x batch_size x emb_dim

        ta_emb_x = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_length)
        ta_emb_x = ta_emb_x.unstack(self.processed_x)

        output_mask = None
        if self.is_training:
            output_mask = make_mask(self.batch_size, self.gen_vd_keep_prob, self.hidden_dim)

        def _pretrain_recurrence(i, x_t, prev_h_state, g_predictions, a_predictions):
            out, state = self.g_recurrent_unit(x_t, prev_h_state)
            out = self.attention_construct_fn(out, self.attention_keys, self.attention_values)

            if output_mask is not None:
                out *= output_mask

            o_t = self.g_output_unit(out)  # batch x vocab , logits not prob
            a_t = self.g_act_unit(out)

            g_predictions = g_predictions.write(i, tf.nn.softmax(o_t))  # batch x vocab_size
            a_predictions = a_predictions.write(i, tf.nn.softmax(a_t))
            x_tp1 = ta_emb_x.read(i)
            return i + 1, x_tp1, state, g_predictions, a_predictions

        _, _, _, self.g_predictions, self.a_predictions = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3, _4: i < self.sequence_length,
            body=_pretrain_recurrence,
            loop_vars=(tf.constant(0, dtype=tf.int32),
                       tf.nn.embedding_lookup(self.g_embeddings, self.start_token),
                       self.h0, g_predictions, a_predictions))

        self.g_predictions = self.g_predictions.stack()
        self.g_predictions = tf.transpose(self.g_predictions, perm=[1, 0, 2])  # batch_size x seq_length x vocab_size

        self.a_predictions = self.a_predictions.stack()
        self.a_predictions = tf.transpose(self.a_predictions, perm=[1, 0, 2])  # batch_size x seq_length x n_actions

        print(self.g_predictions.shape)
        print(self.a_predictions.shape)

    def create_pretrain_loss(self):
        pretrain_recon_loss = self.loss_for_reconstruction()
        pretrain_act_loss = -tf.reduce_sum(
            tf.one_hot(tf.cast(tf.reshape(self.acts, [-1]), tf.int32), self.n_actions, 1.0, 0.0) * clip_and_log(
                tf.reshape(self.a_predictions, [-1, self.n_actions])
            )
        ) / (self.sequence_length * self.batch_size)

        # training updates
        pretrain_loss = pretrain_recon_loss + pretrain_act_loss
        train_op = get_train_op(self.learning_rate, pretrain_loss, self.g_params)
        return pretrain_loss, pretrain_act_loss, train_op

    def create_adversarial_loss(self, dis_predictions):
        missing = get_mask(self.gen_act)

        # mask_sent = tf.cast(get_mask_for_pad(self.gen_x, self.gen_act), tf.float32)
        mask_sent = tf.ones_like(missing, tf.float32)

        rewards = tf.nn.sigmoid(dis_predictions)
        rewards = clip_and_log(rewards)

        missing = tf.cast(missing, tf.float32)
        present = (1 - missing)

        mask_act_log_probs = clip_and_log(self.gen_mask_act_p)
        util_act_log_probs = clip_and_log(1 - self.gen_mask_act_p)

        gen_log_p = self.gen_log_p
        gen_act_log_p = self.gen_act_log_p - tf.stop_gradient(util_act_log_probs)

        probs4Tok = tf.exp(gen_log_p)
        probs4Man = tf.exp(gen_act_log_p)

        log_probs = gen_log_p * missing
        act_log_probs = gen_act_log_p * present

        rewards_list = tf.unstack(rewards, axis=1)
        missing_list = tf.unstack(missing, axis=1)
        present_list = tf.unstack(present, axis=1)
        mask_sent_list = tf.unstack(mask_sent, axis=1)

        # Cumulative Discounted Returns.  The true value function V*(s).
        cumulative_rewards = []
        for t in range(self.sequence_length):
            cum_value = tf.zeros(shape=[self.batch_size])
            for s in range(t, self.sequence_length):
                cum_value += mask_sent_list[s] * np.power(self.reward_gamma, (s - t)) * rewards_list[s]
            cumulative_rewards.append(cum_value)
        cumulative_rewards = tf.stack(cumulative_rewards, axis=1)

        self.critic_loss, self.critic_updates = self.critic.create_critic_loss(cumulative_rewards, missing=mask_sent)

        baselines = tf.unstack(self.critic.estimated_values, axis=1)
        probs4Tok_list = tf.unstack(probs4Tok, axis=1)
        probs4Man_list = tf.unstack(probs4Man, axis=1)
        log_probs_list = tf.unstack(log_probs, axis=1)
        act_log_probs_list = tf.unstack(act_log_probs, axis=1)
        mask_act_log_probs_list = tf.unstack(mask_act_log_probs, axis=1)
        util_act_log_probs_list = tf.unstack(util_act_log_probs, axis=1)

        g_loss = 0.
        for t in range(self.sequence_length):
            prob = probs4Tok_list[t]
            act_prob = probs4Man_list[t]

            log_probability = log_probs_list[t]
            act_log_probability = act_log_probs_list[t]

            mask_log_prob = mask_act_log_probs_list[t]
            util_log_prob = util_act_log_probs_list[t]

            mask_prob = tf.exp(mask_log_prob)
            util_prob = tf.exp(util_log_prob)

            cum_advantage = tf.zeros(shape=[self.batch_size])
            for s in range(t, self.sequence_length):
                cum_advantage += mask_sent_list[s] * np.power(self.reward_gamma, (s - t)) * rewards_list[s]
            cum_advantage -= baselines[t]

            # Clip advantages.
            cum_advantage = tf.clip_by_value(cum_advantage, -self.clip_val, self.clip_val)

            g_loss += tf.multiply(missing_list[t] * log_probability * tf.stop_gradient(mask_prob), tf.stop_gradient(cum_advantage))
            g_loss += tf.multiply(present_list[t] * act_log_probability * tf.stop_gradient(util_prob), tf.stop_gradient(cum_advantage))
            g_loss += tf.multiply(missing_list[t] * mask_log_prob * tf.stop_gradient(prob), tf.stop_gradient(cum_advantage))
            g_loss += tf.multiply(present_list[t] * util_log_prob * tf.stop_gradient(act_prob), tf.stop_gradient(cum_advantage))

        train_op = get_train_op(self.learning_rate, -g_loss, self.g_params, self.clip_val)
        return g_loss, train_op

    def loss_for_reconstruction(self):
        losses = []

        masks = tf.cast(get_mask(self.acts), tf.float32)
        for i in range(self.batch_size):
            target = self.x[i]
            prediction = self.g_predictions[i]

            mask = masks[i]

            one_hot_target = tf.one_hot(tf.cast(tf.reshape(target, [-1]), tf.int32), self.num_vocabulary, 1.0, 0.0)
            log_prob = clip_and_log(tf.reshape(prediction, [-1, self.num_vocabulary]))
            loss = -tf.reduce_sum(one_hot_target * log_prob, -1)
            losses.append(tf.reduce_sum(loss * mask) / tf.reduce_sum(mask))
        return tf.reduce_mean(losses)

    def init_matrix(self, shape):
        return tf.random_normal(shape, stddev=0.1)

    def create_output_unit(self):
        self.Wo = tf.Variable(self.init_matrix([self.hidden_dim, self.num_vocabulary]), name='Wo')
        self.bo = tf.Variable(self.init_matrix([self.num_vocabulary]), name='bo')

        def unit(hidden_state):
            logits = tf.matmul(hidden_state, self.Wo) + self.bo
            return logits

        return unit

    def create_action_unit(self):
        self.Wo_act = tf.Variable(self.init_matrix([self.hidden_dim, self.n_actions]), name='Wo_act')
        self.bo_act = tf.Variable(self.init_matrix([self.n_actions]), name='bo_act')

        def unit(hidden_state):
            logits = tf.matmul(hidden_state, self.Wo_act + self.bo_act)
            return logits

        return unit


if __name__ == '__main__':
    tf.reset_default_graph()
    batch_size, emb_dim, hidden_dim = 2, 3, 3
    sequence_length, seed_length, n_actions = 10, 5, 5
    generator = TokManGAN(100, batch_size, emb_dim, hidden_dim,
                          sequence_length, seed_length, n_actions, is_training=True)

    for var in tf.trainable_variables():
        print(var.op.name)