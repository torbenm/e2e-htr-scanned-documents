import tensorflow as tf
from layer.separable_lstm2d import separable_lstm
from layer.lstm2d import LSTM2D
from layer.rnn2d import multidir_rnn2d, multidir_conv, sum_and_tanh, element_sum, multidir_fullyconnected
from layer.algorithmBase import AlgorithmBase
from layer import resnet
from util import wrap_1d, wrap_4d, make_sparse


def rec_block(net, index, is_train, scope):
    with tf.variable_scope(scope):
        net = wrap_1d(tf.layers.dropout(net, 0.5, training=is_train))
        cell = tf.nn.rnn_cell.LSTMCell(256)
        output, _ = tf.nn.bidirectional_dynamic_rnn(
            cell, cell, net, dtype=tf.float32)
        net = wrap_1d(tf.concat(output, 2))
        return net


class ResHtr(AlgorithmBase):

    def build_graph(self, image_width=200, image_height=100, batch_size=32, channels=1, vocab_length=62, sequence_length=100, learning_rate=0.001):
        x = wrap_1d(tf.placeholder(
            tf.float32, [None, image_width, image_height, channels], name="x"))
        y = tf.sparse_placeholder(
            tf.int32, shape=[None, sequence_length], name="y")
        l = tf.placeholder(
            tf.int32, shape=[None], name="y")
        is_train = tf.placeholder_with_default(False, (), name='is_train')

        num_res = 3
        num_lstm = 5

        ResNet = resnet.Model(resnet_size=3, bottleneck=False, num_filters=16, kernel_size=3,
                              conv_stride=1, first_pool_size=2, first_pool_stride=2,
                              block_sizes=[3] * 3, block_strides=[1] * 3, data_format='channels_last')

        net = wrap_1d(ResNet(x, is_train))

        net = wrap_1d(tf.reshape(
            net, [-1, net.shape[1], net.shape[2] * net.shape[3]]))

        for i in range(num_lstm):
            net = rec_block(net, i, is_train, 'lstm-{}'.format(i))

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
