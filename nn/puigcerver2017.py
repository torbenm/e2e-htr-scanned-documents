import tensorflow as tf
from layer.separable_lstm2d import separable_lstm
from layer.lstm2d import LSTM2D
from layer.rnn2d import multidir_rnn2d, multidir_conv, sum_and_tanh, element_sum, multidir_fullyconnected
from layer.algorithmBase import AlgorithmBase
from util import wrap_1d, wrap_4d, make_sparse, valueOr

"""
Puigcerver, Joan. "Are Multidimensional Recurrent Layers Really Necessary for Handwritten Text Recognition?."
"""


class Puigcerver2017(AlgorithmBase):

    def __init__(self, config):
        self.config = config
        # TODO: Better order
        self.num_conv = valueOr(config, 'conv.num', 5)
        self.num_lstm = valueOr(config, 'lstm.num', 5)
        self.lstm_size = valueOr(config, 'lstm.size', 256)
        self.conv_size = valueOr(config, 'conv.size', 16)

    def _conv_block(self, net, index, is_train):
        num_filters = (index + 1) * self.conv_size
        net = wrap_1d(tf.layers.conv2d(
            net, num_filters, (3, 3), strides=(1, 1)))
        if index > 1:
            net = wrap_1d(tf.layers.dropout(net, 0.2, training=is_train))
        # missing: dropout for layer 3,4,5 (0.2 prob)
        net = wrap_1d(tf.layers.batch_normalization(net, training=is_train))

        # maxpool or dropout first?
        if index < 3:
            net = wrap_1d(tf.layers.max_pooling2d(net, (2, 2), (2, 2)))
        return net

    def _rec_block(self, net, index, is_train, scope):
        with tf.variable_scope(scope):
            net = wrap_1d(tf.layers.dropout(net, 0.5, training=is_train))
            cell = tf.nn.rnn_cell.LSTMCell(self.lstm_size)
            output, _ = tf.nn.bidirectional_dynamic_rnn(
                cell, cell, net, dtype=tf.float32)
            net = wrap_1d(tf.concat(output, 2))
            return net

    def build_graph(self, image_width=200, image_height=100, batch_size=32, channels=1, vocab_length=62, sequence_length=100, learning_rate=0.001):
        x = wrap_1d(tf.placeholder(
            tf.float32, [None, image_width, image_height, channels], name="x"))
        y = tf.sparse_placeholder(
            tf.int32, shape=[None, sequence_length], name="y")
        l = tf.placeholder(
            tf.int32, shape=[None], name="y")
        is_train = tf.placeholder_with_default(False, (), name='is_train')

        net = x
        for i in range(self.num_conv):
            net = self._conv_block(net, i, is_train)

        net = wrap_1d(tf.reshape(
            net, [-1, net.shape[1], net.shape[2] * net.shape[3]]))

        for i in range(self.num_lstm):
            net = self._rec_block(net, i, is_train, 'lstm-{}'.format(i))

        net = wrap_1d(tf.layers.dropout(net, 0.5, training=is_train))
        net = wrap_1d(tf.contrib.layers.fully_connected(net, vocab_length))

        logits = wrap_1d(tf.transpose(net, [1, 0, 2]))
        total_loss = tf.nn.ctc_loss(y, logits, l)

        logits = tf.nn.softmax(logits)
        train_step = tf.train.AdamOptimizer(
            learning_rate).minimize(total_loss)

        return dict(
            x=x,
            y=y,
            l=l,
            is_train=is_train,
            logits=logits,
            total_loss=total_loss,
            train_step=train_step
        )
