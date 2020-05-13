import tensorflow as tf
from tensorflow.python.ops import tensor_array_ops, control_flow_ops
import numpy as np

import common.attention_utils as attention_utils
from common.utils import make_mask, clip_and_log, get_train_op
from common.modules import Generator

from maskgan.modules import Discriminator, Critic


def transform_input_with_is_missing_token(generator, inputs, targets_present):
  input_missing = tf.constant(generator.num_vocabulary, dtype=tf.int32, shape=[generator.batch_size, generator.sequence_length])
  transformed_input = tf.where(tf.cast(targets_present, bool), inputs, input_missing)
  return transformed_input


class MaskGAN(Generator):
    def __init__(self, num_vocabulary, batch_size, emb_dim, hidden_dim, sequence_length,
                 reward_gamma=0.9, is_training=False, gen_vd_keep_prob=1, dis_vd_keep_prob=1, n_rnn_layers=2,
                 clip_val=5.0):

        super().__init__(num_vocabulary, batch_size, emb_dim, hidden_dim, sequence_length, is_training,
                         gen_vd_keep_prob=gen_vd_keep_prob, n_rnn_layers=n_rnn_layers, clip_val=clip_val)

        self.reward_gamma = reward_gamma

        self.dis_vd_keep_prob = dis_vd_keep_prob

        self.is_training = is_training

        # additional placeholder definition
        self.missing = tf.placeholder(tf.int32, shape=[self.batch_size, self.sequence_length], name="missing")
        self.masked_inputs = transform_input_with_is_missing_token(self, self.x, (1 - self.missing))

        with tf.variable_scope('MaskGAN'):
            self.create()
            self.create_network()
            if self.is_training:
                self.create_pretrain_network()

        self.g_params = [var for var in tf.trainable_variables() if var.name.startswith('MaskGAN')]

        if self.is_training:
            self.discriminator = Discriminator(self)
            real_predictions = self.discriminator.create_network(self.x, self.masked_inputs, reuse=False)
            fake_predictions = self.discriminator.create_network(self.gen_x, self.masked_inputs, reuse=True)

            self.dis_loss_fake, self.dis_loss_real, self.dis_updates = \
                self.discriminator.create_loss(fake_predictions, real_predictions, self.missing)

            self.critic = Critic(self, self.g_embeddings, self.gen_x, name='critic')
            self.pretrain_loss, self.pretrain_updates = self.create_pretrain_loss()
            self.g_loss, self.g_updates = self.create_adversarial_loss(fake_predictions)
            self.dis_updates = tf.group(self.dis_updates, self.critic_updates)

    def create(self, reuse=None):
        g_embeddings = self.create_embedding()
        init_missing_embedding = tf.random_uniform([1, self.emb_dim], -1.0, 1.0)
        missing_embedding = tf.get_variable("missing_embedding", initializer=init_missing_embedding)
        self.g_embeddings = tf.concat([g_embeddings, missing_embedding], axis=0)

        attn_cell = self.create_cell(reuse=reuse)
        with tf.variable_scope('seed_encoder') as scope:
            embedded_seed = tf.nn.embedding_lookup(self.g_embeddings, self.masked_inputs)

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
            self.g_recurrent_unit = tf.contrib.rnn.MultiRNNCell([attn_cell() for _ in range(self.n_rnn_layers)], state_is_tuple=True)
            self.g_output_unit = self.create_output_unit()  # maps h_t to o_t (output token logits)


    def create_network(self):
        gen_log_p = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_length, dynamic_size=False,
                                                 infer_shape=True)
        gen_x = tensor_array_ops.TensorArray(dtype=tf.int32, size=self.sequence_length, dynamic_size=False,
                                             infer_shape=True)

        # processed for batch
        with tf.device("/cpu:0"):
            self.processed_missing = tf.transpose(self.missing, perm=[1, 0])
            self.processed_x = tf.transpose(self.x, perm=[1, 0])  # seq_length x batch_size

        ta_emb_x = tensor_array_ops.TensorArray(dtype=tf.int32, size=self.sequence_length)
        ta_emb_x = ta_emb_x.unstack(self.processed_x)

        ta_emb_missing = tensor_array_ops.TensorArray(dtype=tf.int32, size=self.sequence_length)
        ta_emb_missing = ta_emb_missing.unstack(self.processed_missing)

        output_mask = None
        if self.is_training:
            output_mask = make_mask(self.batch_size, self.gen_vd_keep_prob, self.hidden_dim)

        def _g_recurrence(i, x_t, prev_h_state, gen_log_p, gen_x):
            out, state = self.g_recurrent_unit(x_t, prev_h_state)  # hidden_memory_tuple
            out = self.attention_construct_fn(out, self.attention_keys, self.attention_values)

            if output_mask is not None:
                out *= output_mask

            if not self.is_training:
                out *= self.temp

            o_t = self.g_output_unit(out)  # batch x vocab , logits not prob

            prob = tf.nn.softmax(o_t)
            log_prob = clip_and_log(prob)

            next_token = tf.squeeze(tf.random.categorical(log_prob, 1))
            next_token = tf.cast(next_token, tf.int32)

            is_present = 1 - ta_emb_missing.read(i)
            given_next_token = ta_emb_x.read(i)

            next_token = tf.where(tf.cast(is_present, tf.bool), given_next_token, next_token)

            x_tp1 = tf.nn.embedding_lookup(self.g_embeddings, next_token)  # batch x emb_dim

            one_hot_next_token = tf.one_hot(next_token, self.num_vocabulary, 1.0, 0.0)
            log_p_next_token = tf.multiply(one_hot_next_token, log_prob)

            gen_log_p = gen_log_p.write(i, tf.reduce_sum(log_p_next_token, 1))  # [batch_size] , prob
            gen_x = gen_x.write(i, next_token)  # indices, batch_size
            return i + 1, x_tp1, state, gen_log_p, gen_x

        _, _, _, self.gen_log_p, self.gen_x = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3, _4: i < self.sequence_length,
            body=_g_recurrence,
            loop_vars=(tf.constant(0, dtype=tf.int32),
                       tf.nn.embedding_lookup(self.g_embeddings, self.start_token),
                       self.h0, gen_log_p, gen_x))

        self.gen_x = self.gen_x.stack()  # seq_length x batch_size
        self.gen_x = tf.transpose(self.gen_x, perm=[1, 0])  # batch_size x seq_length
        self.gen_x = tf.reshape(self.gen_x, shape=[self.batch_size, self.sequence_length])

        self.gen_log_p = self.gen_log_p.stack()  # seq_length x batch_size
        self.gen_log_p = tf.transpose(self.gen_log_p, perm=[1, 0])  # batch_size x seq_length
        self.gen_log_p = tf.reshape(self.gen_log_p, shape=[self.batch_size, self.sequence_length])

    def create_pretrain_network(self):
        with tf.variable_scope('attention'):
            (self.attention_keys, self.attention_values, _,
             self.attention_construct_fn) = attention_utils.prepare_attention(
                self.encoded_seed, 'luong', num_units=self.hidden_dim, reuse=True)

        # supervised pretraining for generator
        g_predictions = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_length, dynamic_size=False,
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

        def _pretrain_recurrence(i, x_t, prev_h_state, g_predictions):
            out, state = self.g_recurrent_unit(x_t, prev_h_state)
            out = self.attention_construct_fn(out, self.attention_keys, self.attention_values)

            if output_mask is not None:
                out *= output_mask

            o_t = self.g_output_unit(out)  # batch x vocab , logits not prob

            g_predictions = g_predictions.write(i, tf.nn.softmax(o_t))  # batch x vocab_size

            x_tp1 = ta_emb_x.read(i)
            return i + 1, x_tp1, state, g_predictions

        _, _, _, self.g_predictions = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3: i < self.sequence_length,
            body=_pretrain_recurrence,
            loop_vars=(tf.constant(0, dtype=tf.int32),
                       tf.nn.embedding_lookup(self.g_embeddings, self.start_token),
                       self.h0, g_predictions))

        self.g_predictions = self.g_predictions.stack()
        self.g_predictions = tf.transpose(self.g_predictions, perm=[1, 0, 2])  # batch_size x seq_length x vocab_size

        print(self.g_predictions.shape)

    def create_pretrain_loss(self):
        pretrain_loss = self.loss_for_reconstruction()

        # training updates
        train_op = get_train_op(self.learning_rate, pretrain_loss, self.g_params, self.clip_val)
        return pretrain_loss, train_op

    def create_adversarial_loss(self, dis_predictions):
        missing = tf.cast(self.missing, tf.float32)

        rewards = tf.nn.sigmoid(dis_predictions)
        rewards = clip_and_log(rewards)

        log_probs = self.gen_log_p * missing

        rewards_list = tf.unstack(rewards, axis=1)
        missing_list = tf.unstack(missing, axis=1)

        # Cumulative Discounted Returns.  The true value function V*(s).
        cumulative_rewards = []
        for t in range(self.sequence_length):
            cum_value = tf.zeros(shape=[self.batch_size])
            for s in range(t, self.sequence_length):
                cum_value += missing_list[s] * np.power(self.reward_gamma, (s - t)) * rewards_list[s]
            cumulative_rewards.append(cum_value)
        cumulative_rewards = tf.stack(cumulative_rewards, axis=1)
        print("cumulative_rewards:", cumulative_rewards.shape)

        # Unstack Tensors into lists.
        self.critic_loss, self.critic_updates = self.critic.create_critic_loss(cumulative_rewards, self.missing)

        baselines = tf.unstack(self.critic.estimated_values, axis=1)
        log_probs_list = tf.unstack(log_probs, axis=1)

        g_loss = 0.
        for t in range(self.sequence_length):
            log_probability = log_probs_list[t]

            cum_advantage = tf.zeros(shape=[self.batch_size])
            for s in range(t, self.sequence_length):
                cum_advantage += missing_list[s] * np.power(self.reward_gamma, (s - t)) * rewards_list[s]
            cum_advantage -= baselines[t]

            # Clip advantages.
            cum_advantage = tf.clip_by_value(cum_advantage, -self.clip_val, self.clip_val)
            g_loss += tf.multiply(missing_list[t] * log_probability, tf.stop_gradient(cum_advantage))

        train_op = get_train_op(self.learning_rate, -g_loss, self.g_params, self.clip_val)
        return g_loss, train_op

    def loss_for_reconstruction(self):
        losses = []

        for i in range(self.batch_size):
            target = self.x[i]
            prediction = self.g_predictions[i]

            mask = tf.cast(self.missing[i], tf.float32)

            one_hot_target = tf.one_hot(tf.cast(tf.reshape(target, [-1]), tf.int32), self.num_vocabulary, 1.0, 0.0)
            log_prob = clip_and_log(tf.reshape(prediction, [-1, self.num_vocabulary]))
            loss = -tf.reduce_sum(one_hot_target * log_prob, -1)
            losses.append(tf.reduce_sum(loss * mask) / tf.reduce_sum(mask))
        return tf.reduce_mean(losses)


if __name__ == '__main__':
    tf.reset_default_graph()
    batch_size, emb_dim, hidden_dim = 2, 3, 3
    sequence_length = 10
    generator = MaskGAN(100, batch_size, emb_dim, hidden_dim, sequence_length, is_training=True)

    for var in tf.trainable_variables():
        print(var.op.name)