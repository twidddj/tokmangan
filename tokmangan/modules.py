import tensorflow as tf
from common.config import ACT
from common.utils import VariationalDropoutWrapper, make_mask, get_train_op
from common.modules import dis_encoder, dis_decoder


class Discriminator(object):
    def __init__(self, generator):
        self.generator = generator

    def create_network(self, gen_sequence, transformed_seed, reuse=None):
        with tf.variable_scope('TokGANGenerator', reuse=True):
            d_embedding = tf.get_variable('embeddings', [self.generator.num_vocabulary, self.generator.emb_dim])

        with tf.variable_scope('discriminator', reuse=reuse):
            # add, replace, ignore, term
            man_token_embedding = tf.get_variable('token_embedding', [4, self.generator.emb_dim])
            e_embedding = tf.concat([d_embedding, man_token_embedding], axis=0)

            encoder_states = dis_encoder(self.generator, e_embedding, transformed_seed, reuse=reuse)
            predictions = dis_decoder(self.generator, d_embedding, gen_sequence, encoder_states, reuse=reuse)
        return predictions

    def create_loss(self, fake_predictions, real_predictions,
                    fake_sequence, real_sequence,
                    fake_missing, real_missing,
                    fake_weights=1.0, real_weights=1.0):

        real_labels = tf.ones_like(real_predictions)
        fake_labels = tf.zeros_like(fake_predictions)

        real_presented = tf.cast(real_sequence, tf.float32) * (1 - real_missing)
        fake_presented = tf.cast(fake_sequence, tf.float32) * (1 - fake_missing)

        # all presented tokens > 0 but missing = -1
        _fake_presented = tf.where(tf.math.equal(fake_presented, 0), tf.ones_like(fake_presented) * -1, fake_presented)
        fake_labels = tf.where(tf.math.equal(real_presented, _fake_presented), real_labels, fake_labels)

        loss_real = tf.compat.v1.losses.sigmoid_cross_entropy(real_labels, real_predictions, weights=real_weights)
        loss_fake = tf.compat.v1.losses.sigmoid_cross_entropy(fake_labels, fake_predictions, weights=fake_weights)

        loss = (loss_fake + loss_real) / 2.

        vars = [param for param in tf.trainable_variables() if 'discriminator' in param.name]
        train_op = get_train_op(self.generator.learning_rate, loss, vars)

        return loss_fake, loss_real, train_op


class Critic(object):
    def __init__(self, generator, embedding, xs, acts, reuse=None, num_layers=2, name='critic'):
        self.generator = generator
        self.name = name
        cell_critic, dis_scope = self.create_lstm_cell(num_layers)

        with tf.variable_scope(name, reuse=reuse):
            man_token_embedding = tf.get_variable('token_embedding', [len(ACT), self.generator.emb_dim])
            state_critic = cell_critic.zero_state(generator.batch_size, tf.float32)
            if generator.is_training:
                output_mask = make_mask(generator.batch_size, generator.dis_vd_keep_prob, generator.hidden_dim)

            with tf.variable_scope('rnn') as vs:
                values = []
                rnn_inputs = tf.nn.embedding_lookup(embedding, xs)
                rnn_man_inputs = tf.nn.embedding_lookup(man_token_embedding, acts)

                for t in range(generator.sequence_length):
                    if t > 0:
                        tf.get_variable_scope().reuse_variables()

                    rnn_in = rnn_inputs[:, t] + rnn_man_inputs[:, t]
                    rnn_out, state_critic = cell_critic(rnn_in, state_critic, scope=dis_scope)

                    if generator.is_training:
                        rnn_out *= output_mask

                    value = tf.contrib.layers.linear(rnn_out, 1, scope=vs)
                    values.append(value)

        self.estimated_values = tf.squeeze(tf.stack(values, axis=1), -1)

    def create_lstm_cell(self, num_layers):
        with tf.variable_scope('discriminator/decoder/rnn/multi_rnn_cell', reuse=True) as dis_scope:
            def lstm_cell():
                return tf.contrib.rnn.BasicLSTMCell(self.generator.hidden_dim, forget_bias=0.0, state_is_tuple=True, reuse=True)

            attn_cell = lstm_cell
            if self.generator.is_training and self.generator.dis_vd_keep_prob < 1:
                def attn_cell():
                    return VariationalDropoutWrapper(lstm_cell(), self.generator.batch_size, self.generator.dis_vd_keep_prob, self.generator.dis_vd_keep_prob)

            cell_critic = tf.contrib.rnn.MultiRNNCell([attn_cell() for _ in range(num_layers)], state_is_tuple=True)
            return cell_critic, dis_scope

    def create_critic_loss(self, cumulative_rewards, missing=None):
        if missing is not None:
            missing = tf.cast(missing, tf.bool)
        else:
            missing = 1.0

        loss = tf.compat.v1.losses.mean_squared_error(labels=cumulative_rewards, predictions=self.estimated_values, weights=missing)
        vars = [v for v in tf.trainable_variables() if v.op.name.startswith(self.name)]
        train_op = get_train_op(self.generator.learning_rate, loss, vars)

        return loss, train_op