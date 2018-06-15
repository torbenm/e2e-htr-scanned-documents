import tensorflow as tf
from .layer.separable_lstm2d import separable_lstm
from .layer.lstm2d import LSTM2D
from .layer.rnn2d import multidir_rnn2d, multidir_conv, sum_and_tanh, element_sum, multidir_fullyconnected
from .layer.algorithmBase import AlgorithmBase
from .util import wrap_1d, wrap_4d, make_sparse

"""
Graves, Alex, and Juergen Schmidhuber. "Offline handwriting recognition
with multidimensional recurrent neural networks." Advances in neural
information processing systems. 2009.
"""

SEPARABLE = False


def lstm_conv_layer(x, lstm_size, lstm_shape, kernel_size, filter, name):
    if not SEPARABLE:
        lstm_cell = LSTM2D(lstm_size)
        net = wrap_4d(multidir_rnn2d(
            lstm_cell, x, sequence_shape=lstm_shape, scope=name, dtype=tf.float32))
        net = wrap_4d(multidir_conv(net, kernel_size=kernel_size,
                                    strides=kernel_size, filters=filter))
        return wrap_1d(sum_and_tanh(net))
    else:
        net = wrap_1d(separable_lstm(
            x, lstm_size, kernel_size=[lstm_shape[1], lstm_shape[0]], scope=name))
        net = wrap_1d(tf.layers.conv2d(
            net, filter, kernel_size, strides=kernel_size, activation=tf.tanh))
        return net


def lstm_fc_layer(net, lstm_size, lstm_shape, fc_units, name):
    if not SEPARABLE:
        lstm_cell = LSTM2D(lstm_size)
        net = wrap_4d(multidir_rnn2d(
            lstm_cell, net, sequence_shape=lstm_shape, scope=name, dtype=tf.float32))  # is this really the sequence shape?
        net = wrap_4d(multidir_fullyconnected(net, units=fc_units))
        net = wrap_1d(element_sum(net, axis=[0, 3]))
    else:
        net = wrap_1d(separable_lstm(
            net, lstm_size, kernel_size=[lstm_shape[1], lstm_shape[0]], scope=name))
        net = tf.contrib.layers.fully_connected(
            inputs=net, num_outputs=fc_units)
        net = wrap_1d(element_sum(net, axis=[2]))
    return net


class GravesSchmidhuber2009(AlgorithmBase):

    def __init__(self, config):
        self.config = config

    def build_graph(self, image_width=200, image_height=100, batch_size=32, channels=1, vocab_length=62, sequence_length=100, learning_rate=0.001):

        x = wrap_1d(tf.placeholder(
            tf.float32, [batch_size, image_width, image_height, channels], name="x"))
        y = tf.sparse_placeholder(
            tf.int32, shape=[batch_size, sequence_length], name="y")
        l = tf.placeholder(
            tf.int32, shape=[batch_size], name="l")

        # MDLSTM Layer 1
        net = lstm_conv_layer(x, 2, (1, 1), (2, 2), 6, 'lstm-1')
        net = lstm_conv_layer(net, 10, (1, 1), (2, 2), 20, 'lstm-2')

        # MDLSTM layer 3
        net = lstm_fc_layer(net, 50, (1, 1), vocab_length, 'lstm-3')
        # net = tf.nn.softmax(net)

        logits = wrap_1d(tf.transpose(net, [1, 0, 2]))

        total_loss = tf.nn.ctc_loss(y, logits, l)

        logits = tf.nn.softmax(logits)
        decoded, _ = tf.nn.ctc_greedy_decoder(
            logits, l, merge_repeated=True)
        # wrap_1d(decoded[0])
        # ler = tf.reduce_mean(tf.edit_distance(
        #     decoded[0], tf.cast(y, tf.int64)))

        # decoded = tf.cast(decoded[0], tf.int32)
        decoded = tf.sparse_to_dense(
            decoded[0].indices, decoded[0].dense_shape, decoded[0].values, tf.constant(vocab_length - 1, dtype=tf.int64))
        train_step = tf.train.AdamOptimizer(
            learning_rate).minimize(total_loss)

        return dict(
            x=x,
            y=y,
            l=l,
            # ler=ler,
            logits=logits,
            output=decoded,
            total_loss=total_loss,
            train_step=train_step,
            saver=tf.train.Saver()
        )
