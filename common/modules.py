import tensorflow as tf
from common.utils import VariationalDropoutWrapper, make_mask
import common.attention_utils as attention_utils


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