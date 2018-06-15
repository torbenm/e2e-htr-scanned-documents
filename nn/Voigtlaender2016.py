import tensorflow as tf
from .layer.separable_lstm2d import separable_lstm
from .layer.lstm2d import LSTM2D
from .layer.rnn2d import multidir_rnn2d, multidir_conv, sum_and_tanh, element_sum, multidir_fullyconnected
from .layer.mdlstm2d import mdlstm2d
from .layer.algorithmBase import AlgorithmBase
from .util import wrap_1d, wrap_4d, make_sparse

"""
Voigtlaender, Paul, Patrick Doetsch, and Hermann Ney. 
"Handwriting recognition with large multidimensional long short-term memory recurrent neural networks." 
Frontiers in Handwriting Recognition (ICFHR), 2016 15th International Conference on. IEEE, 2016.
"""


def conv_mdlstm_block(net, idx, is_train, width=5, dropout=True):
    real_idx = (idx * 2) + 1
    if dropout:
        net = wrap_1d(tf.layers.dropout(net, 0.25, training=is_train))
    net = wrap_1d(tf.layers.conv2d(
        net, width * real_idx, (3, 3), activation=tf.tanh))
    net = wrap_1d(tf.layers.max_pooling2d(net, (2, 2), (2, 2)))

    net = wrap_1d(tf.layers.dropout(net, 0.25, training=is_train))

    # ---- START LSTM
    net = mdlstm2d((real_idx + 1) * width, net, scope='lstm-{}'.format(idx))
    # ---- END LSTM

    return net


class VoigtlaenderDoetschNey2016(AlgorithmBase):

    def __init__(self, config):
        self.config = config

    def build_graph(self, image_width=200, image_height=100, batch_size=32, channels=1, vocab_length=62, sequence_length=100, learning_rate=0.001):
        x = wrap_1d(tf.placeholder(
            tf.float32, [batch_size, image_width, image_height, channels], name="x"))
        y = tf.sparse_placeholder(
            tf.int32, shape=[batch_size, sequence_length], name="y")
        l = tf.placeholder(
            tf.int32, shape=[batch_size], name="y")
        is_train = tf.placeholder_with_default(False, (), name='is_train')

        num_layers = 3
        width = 5

        net = x

        for i in range(num_layers):
            net = conv_mdlstm_block(net, i, is_train, width, i != 0)

        net = wrap_1d(element_sum(net, axis=[2]))
        logits = wrap_1d(tf.transpose(net, [1, 0, 2]))

        total_loss = tf.nn.ctc_loss(y, logits, l)
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
