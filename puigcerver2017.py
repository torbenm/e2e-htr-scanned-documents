import tensorflow as tf
from layer.separable_lstm2d import separable_lstm
from layer.lstm2d import LSTM2D
from layer.rnn2d import multidir_rnn2d, multidir_conv, sum_and_tanh, element_sum, multidir_fullyconnected
from layer.algorithmBase import AlgorithmBase
from util import wrap_1d, wrap_4d, make_sparse


def conv_block(net, index):
    num_filters = (index + 1) * 16
    net = wrap_1d(tf.layers.conv2d(
        net, num_filters, (3, 3), strides=(1, 1), activation=tf.nn.leaky_relu))
    if index > 1:
        net = wrap_1d(tf.nn.dropout(net, 0.8))
    # missing: dropout for layer 3,4,5 (0.2 prob)
    net = wrap_1d(tf.layers.batch_normalization(net))

    # maxpool or dropout first?
    if index < 3:
        net = wrap_1d(tf.layers.max_pooling2d(net, (2, 2), (2, 2)))
    return net


def rec_block(net, index, scope):
    with tf.variable_scope(scope):
        net = wrap_1d(tf.nn.dropout(net, 0.5))
        cell = tf.nn.rnn_cell.LSTMCell(256)
        output, _ = tf.nn.bidirectional_dynamic_rnn(
            cell, cell, net, dtype=tf.float32)
        net = wrap_1d(tf.concat(output, 2))
        return net


class Puigcerver2017(AlgorithmBase):

    def build_graph(self, image_width=200, image_height=100, batch_size=32, channels=1, vocab_length=62, sequence_length=100, learning_rate=0.001):
        x = wrap_1d(tf.placeholder(
            tf.float32, [batch_size, image_width, image_height, channels], name="x"))
        y = tf.sparse_placeholder(
            tf.int32, shape=[batch_size, sequence_length], name="y")
        l = tf.placeholder(
            tf.int32, shape=[batch_size], name="y")

        num_conv = 5
        num_lstm = 5

        net = x
        for i in range(num_conv):
            net = conv_block(net, i)

        net = wrap_1d(tf.reshape(net, [batch_size, net.shape[1], -1]))

        for i in range(num_lstm):
            net = rec_block(net, i, 'lstm-{}'.format(i))

        net = wrap_1d(tf.nn.dropout(net, 0.5))
        net = wrap_1d(tf.contrib.layers.fully_connected(net, vocab_length))

        logits = wrap_1d(tf.transpose(net, [1, 0, 2]))
        total_loss = tf.nn.ctc_loss(y, logits, l)

        logits = tf.nn.softmax(logits)
        decoded, _ = tf.nn.ctc_greedy_decoder(logits, l, merge_repeated=True)
        # wrap_1d(decoded[0])

        # decoded = tf.cast(decoded[0], tf.int32)
        decoded = tf.sparse_to_dense(
            decoded[0].indices, decoded[0].dense_shape, decoded[0].values)
        train_step = tf.train.AdamOptimizer(
            learning_rate).minimize(total_loss)

        return dict(
            x=x,
            y=y,
            l=l,
            output=decoded,
            total_loss=total_loss,
            train_step=train_step,
            saver=tf.train.Saver()
        )
