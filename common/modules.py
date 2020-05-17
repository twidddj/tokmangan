import tensorflow as tf
from common.utils import VariationalDropoutWrapper, make_mask, clip_and_log, get_train_op
import common.attention_utils as attention_utils
from common.config import START_TOKEN_IDX, END_TOKEN_IDX, PAD_TOKEN_IDX
from tensorflow.python.ops import tensor_array_ops, control_flow_ops


class Generator(object):
    def __init__(self, num_vocabulary, batch_size, emb_dim, hidden_dim, sequence_length, is_training=False,
                 gen_vd_keep_prob=1, n_rnn_layers=2, clip_val=5.0):
        self.is_training = is_training
        self.num_vocabulary = num_vocabulary
        self.batch_size = batch_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.start_token = tf.constant([START_TOKEN_IDX] * self.batch_size, dtype=tf.int32)
        self.end_token = tf.constant([END_TOKEN_IDX] * self.batch_size, dtype=tf.int32)
        self.pad_token = tf.constant([PAD_TOKEN_IDX] * self.batch_size, dtype=tf.int32)
        self.gen_vd_keep_prob = gen_vd_keep_prob
        self.n_rnn_layers = n_rnn_layers
        self.clip_val = clip_val

        # placeholder definition
        self.x = tf.placeholder(tf.int32, shape=[self.batch_size, self.sequence_length], name="sparse_x")
        self.x_len = tf.placeholder(tf.int32, shape=[self.batch_size, ], name="x_len")  # n_x
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        self.temp = tf.placeholder(tf.float32, shape=[], name='temperature')

    def create(self, reuse=None):
        with tf.variable_scope('LM'):
            self.g_embeddings = self.create_embedding()
            with tf.variable_scope('generator'):
                attn_cell = self.create_cell(cond=True, reuse=reuse)
                self.g_recurrent_unit = tf.contrib.rnn.MultiRNNCell([attn_cell() for _ in range(self.n_rnn_layers)], state_is_tuple=True)
                self.g_output_unit = self.create_output_unit()  # maps h_t to o_t (output token logits)
                self.h0 = self.g_recurrent_unit.zero_state(self.batch_size, tf.float32)

            self.create_network()
            self.create_pretrain_network()


        self.g_params = [var for var in tf.trainable_variables() if var.name.startswith('LM')]
        self.pretrain_loss, self.pretrain_updates = self.create_pretrain_loss()
        self.masked_nlls = self.nll()

    def create_network(self):
        gen_log_p = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_length, dynamic_size=False,
                                                 infer_shape=True)
        gen_x = tensor_array_ops.TensorArray(dtype=tf.int32, size=self.sequence_length, dynamic_size=False,
                                             infer_shape=True)

        output_mask = None
        if self.is_training:
            output_mask = make_mask(self.batch_size, self.gen_vd_keep_prob, self.hidden_dim)

        def _g_recurrence(i, x_t, prev_h_state, gen_log_p, gen_x):
            out, state = self.g_recurrent_unit(x_t, prev_h_state)  # hidden_memory_tuple

            if output_mask is not None:
                out *= output_mask

            if not self.is_training:
                out *= self.temp

            o_t = self.g_output_unit(out)  # batch x vocab , logits not prob

            prob = tf.nn.softmax(o_t)
            log_prob = clip_and_log(prob)

            next_token = tf.squeeze(tf.random.categorical(log_prob, 1))
            next_token = tf.cast(next_token, tf.int32)

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

    def loss_for_reconstruction(self):
        losses = []

        for i in range(self.batch_size):
            target = self.x[i]
            prediction = self.g_predictions[i]

            one_hot_target = tf.one_hot(tf.cast(tf.reshape(target, [-1]), tf.int32), self.num_vocabulary, 1.0, 0.0)
            log_prob = clip_and_log(tf.reshape(prediction, [-1, self.num_vocabulary]))
            loss = -tf.reduce_sum(one_hot_target * log_prob, -1)
            losses.append(tf.reduce_sum(loss) / self.sequence_length)
        return tf.reduce_mean(losses)

    def nll(self):
        losses = []

        masks = tf.sequence_mask(self.x_len, maxlen=self.sequence_length, dtype=tf.float32)
        for i in range(self.batch_size):
            target = self.x[i]
            prediction = self.g_predictions[i]
            mask = masks[i]

            one_hot_target = tf.one_hot(tf.cast(tf.reshape(target, [-1]), tf.int32), self.num_vocabulary, 1.0, 0.0)
            log_prob = clip_and_log(tf.reshape(prediction, [-1, self.num_vocabulary]))
            loss = -tf.reduce_sum(one_hot_target * log_prob, -1)
            losses.append(tf.reduce_sum(loss * mask) / tf.reduce_sum(mask))
        return losses

    def create_embedding(self):
        init_embeddings = tf.random_uniform([self.num_vocabulary, self.emb_dim], -1.0, 1.0)
        g_embeddings = tf.get_variable("embeddings", initializer=init_embeddings)
        return g_embeddings

    def create_cell(self, cond=None, reuse=None):
        if cond is None:
            cond = self.is_training

        def lstm_cell():
            return tf.contrib.rnn.BasicLSTMCell(self.hidden_dim, forget_bias=0.0, state_is_tuple=True, reuse=reuse)

        attn_cell = lstm_cell
        if cond and self.gen_vd_keep_prob < 1:
            def attn_cell():
                return VariationalDropoutWrapper(lstm_cell(), self.batch_size, self.gen_vd_keep_prob,
                                                 self.gen_vd_keep_prob)
        return attn_cell

    def init_matrix(self, shape):
        return tf.random_normal(shape, stddev=0.1)

    def create_output_unit(self):
        self.Wo = tf.Variable(self.init_matrix([self.hidden_dim, self.num_vocabulary]), name='Wo')
        self.bo = tf.Variable(self.init_matrix([self.num_vocabulary]), name='bo')

        def unit(hidden_state):
            logits = tf.matmul(hidden_state, self.Wo) + self.bo
            return logits

        return unit


def dis_encoder(generator, embedding, transformed_inputs, reuse=None, dis_num_layers=2):
    print("transformed_inputs:", transformed_inputs.shape)

    with tf.variable_scope('encoder', reuse=reuse):
        def lstm_cell():
            return tf.contrib.rnn.BasicLSTMCell(generator.hidden_dim, forget_bias=0.0, state_is_tuple=True, reuse=reuse)

        attn_cell = lstm_cell
        if generator.is_training and generator.dis_vd_keep_prob < 1:
            def attn_cell():
                return VariationalDropoutWrapper(lstm_cell(), generator.batch_size, generator.dis_vd_keep_prob, generator.dis_vd_keep_prob)

        cell_dis = tf.contrib.rnn.MultiRNNCell([attn_cell() for _ in range(dis_num_layers)], state_is_tuple=True)

        state_dis = cell_dis.zero_state(generator.batch_size, tf.float32)

        if generator.is_training:
            output_mask = make_mask(generator.batch_size, generator.dis_vd_keep_prob, generator.hidden_dim)

        with tf.variable_scope('rnn'):
            hidden_states = []
            rnn_inputs = tf.nn.embedding_lookup(embedding, transformed_inputs)

            for t in range(generator.sequence_length):
                if t > 0:
                    tf.get_variable_scope().reuse_variables()

                rnn_in = rnn_inputs[:, t]
                rnn_out, state_dis = cell_dis(rnn_in, state_dis)
                if generator.is_training:
                    rnn_out *= output_mask

                hidden_states.append(rnn_out)
                final_state = state_dis

    return tf.stack(hidden_states, axis=1), final_state


def dis_decoder(generator, embedding, sequence, encoding_state, reuse=None, dis_num_layers=2):
    sequence = tf.cast(sequence, tf.int32)

    with tf.variable_scope('decoder', reuse=reuse):
        def lstm_cell():
            return tf.contrib.rnn.BasicLSTMCell(generator.hidden_dim, forget_bias=0.0, state_is_tuple=True, reuse=reuse)

        attn_cell = lstm_cell
        if generator.is_training and generator.dis_vd_keep_prob < 1:
            def attn_cell():
                return VariationalDropoutWrapper(lstm_cell(), generator.batch_size,  generator.dis_vd_keep_prob, generator.dis_vd_keep_prob)

        cell_dis = tf.contrib.rnn.MultiRNNCell([attn_cell() for _ in range(dis_num_layers)], state_is_tuple=True)

        state = encoding_state[1]

        (attention_keys, attention_values, _, attention_construct_fn) = attention_utils.prepare_attention(
            encoding_state[0],
            'luong',
            num_units=generator.hidden_dim,
            reuse=reuse)

        if generator.is_training:
            output_mask = make_mask(generator.batch_size, generator.dis_vd_keep_prob, generator.hidden_dim)

        with tf.variable_scope('rnn') as vs:
            predictions = []

            rnn_inputs = tf.nn.embedding_lookup(embedding, sequence)

            for t in range(generator.sequence_length):
                if t > 0:
                    tf.get_variable_scope().reuse_variables()

                rnn_in = rnn_inputs[:, t]
                rnn_out, state = cell_dis(rnn_in, state)
                rnn_out = attention_construct_fn(rnn_out, attention_keys, attention_values)
                if generator.is_training:
                    rnn_out *= output_mask

                pred = tf.contrib.layers.linear(rnn_out, 1, scope=vs)
                predictions.append(pred)
        predictions = tf.stack(predictions, axis=1)
    return tf.squeeze(predictions, axis=2)


if __name__ == '__main__':
    tf.reset_default_graph()
    batch_size, emb_dim, hidden_dim = 2, 3, 3
    sequence_length = 10
    generator = Generator(100, batch_size, emb_dim, hidden_dim, sequence_length, is_training=False)
    generator.create()
    for var in tf.trainable_variables():
        print(var.op.name)