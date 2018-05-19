import tensorflow as tf
from layer.separable_lstm2d import separable_lstm
from layer.lstm2d import LSTM2D
from layer.rnn2d import multidir_rnn2d, multidir_conv, sum_and_tanh, element_sum, multidir_fullyconnected
from layer.algorithmBase import AlgorithmBase
from util import wrap_1d, wrap_4d, make_sparse


def conv_mdlstm_block(net, idx, width=5, dropout=True):
    real_idx = (idx * 2) + 1
    if dropout:
        net = wrap_1d(tf.nn.dropout(net, 0.75))
    net = wrap_1d(tf.layers.conv2d(
        net, width * real_idx, (3, 3), activation=None))
    net = wrap_1d(tf.layers.max_pooling2d(net, (2, 2), (2, 2)))
    net = wrap_1d(tf.tanh(net))
    net = wrap_1d(tf.nn.dropout(net, 0.75))
    lstm = LSTM2D((real_idx + 1) * width)
    net = wrap_4d(multidir_rnn2d(lstm, net, (1, 1),
                                 dtype=tf.float32, scope='lstm-{}'.format(idx)))
    net = wrap_1d(element_sum(net, reducer=tf.reduce_mean))
    return net


class VoigtlaenderDoetschNey2016(AlgorithmBase):

    def build_graph(self, image_width=200, image_height=100, batch_size=32, channels=1, vocab_length=62, sequence_length=100, learning_rate=0.001):
        x = wrap_1d(tf.placeholder(
            tf.float32, [batch_size, image_width, image_height, channels], name="x"))
        y = tf.sparse_placeholder(
            tf.int32, shape=[batch_size, sequence_length], name="y")
        l = tf.placeholder(
            tf.int32, shape=[batch_size], name="y")

        num_layers = 3
        width = 5

        net = x
        for i in range(num_layers):
            net = conv_mdlstm_block(net, i, width, i != 0)

        net = wrap_1d(element_sum(net, axis=[2]))

        logits = wrap_1d(tf.transpose(net, [1, 0, 2]))
        decoded, _ = tf.nn.ctc_greedy_decoder(logits, l, merge_repeated=False)
        # wrap_1d(decoded[0])

        # decoded = tf.cast(decoded[0], tf.int32)
        decoded = tf.sparse_to_dense(
            decoded[0].indices, decoded[0].dense_shape, decoded[0].values)
        total_loss = tf.nn.ctc_loss(y, logits, l)
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
