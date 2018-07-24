import tensorflow as tf
from nn.layer.algorithmBaseV2 import AlgorithmBaseV2
from nn.util import log_1d
from nn.layer.histogrammed import conv2d, batch_normalization


DEFAULTS = {
    "conv1": {
        "kernel": [3, 3],
        "features": 64
    },
    "convs": {
        "num": 16,
        "kernel": [3, 3],
        "features": 64
    },
    "conv_n": {
        "kernel": [3, 3]
    }
}


class DnCNN(AlgorithmBaseV2):

    def __init__(self, config):
        super(DnCNN, self).__init__(config, DEFAULTS)

    def _conv_bn_layer(self, net, index, train):
        with tf.name_scope('conv{}'.format(index)):
            net = conv2d(net, self['convs.features'], self['convs.kernel'],
                         padding='same', activation=None, use_bias=False)
            net = batch_normalization(net, training=train)
            return tf.nn.relu(net)

    def build_graph(self, channels=1, learning_rate=0.001):

        ###################
        # PLACEHOLDER
        ###################
        with tf.name_scope('placeholder'):
            x = log_1d(tf.placeholder(
                tf.float32, [None, None, None, channels], name="x"))
            y = tf.placeholder(tf.float32, shape=x.shape, name="y")
            is_train = tf.placeholder_with_default(False, (), name='is_train')

            net = x

        ################
        # PHASE I: Convolutional Block without BN
        ###############
        with tf.name_scope('conv1'):
            net = log_1d(conv2d(net, self['conv1.features'], self['conv1.kernel'],
                                padding='same', activation=tf.nn.relu))

        ################
        # PHASE II: Conv Loop
        ###############
        for idx in range(self['convs.num']):
            net = log_1d(self._conv_bn_layer(net, idx+1, is_train))

        ################
        # PHASE III: Last Conv Block
        ###############
        with tf.name_scope('conv_n'):
            output = log_1d(conv2d(net, channels, self['conv_n.kernel']))

        ##################
        # PHASE IV: Loss & Optomizer
        #################
        loss = tf.nn.l2_loss(output - y)
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        return dict(
            x=x,
            y=y,
            is_train=is_train,
            output=output,
            loss=loss,
            train_step=train_step
        )
