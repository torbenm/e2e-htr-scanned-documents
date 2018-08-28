import tensorflow as tf
from nn.layer.algorithmBaseV2 import AlgorithmBaseV2
from nn.util import log_1d
from nn.layer.histogrammed import conv2d, batch_normalization


DEFAULTS = {
    "conv1": {
        "kernel": [3, 3],
        "features": 64
    },
    "downconv": {
        "layers": [False, True, True, True, True],
        "filters": [8, 8]
    },
    "conv_n": {
        "kernel": [3, 3],
        "activation": ""
    }
}


class Unet(AlgorithmBaseV2):

    def __init__(self, config):
        super(Unet, self).__init__(config, DEFAULTS)

    def down_conv(self, net, index, prepool, is_train):
        with tf.variable_scope("down_{}".format(index)):
            if prepool:
                net = tf.layers.max_pooling2d(net, (2, 2), strides=(
                    2, 2), name="prepool_{}".format(index))
            for i, f in enumerate(self['downconv.filters']):
                # regularizer?
                net = conv2d(net, filters ** (index+1),
                             activation=None, name="conv_{}".format(i))
                net = batch_normalization(net, training=train)
                net = tf.nn.relu(net)
        return net

    def up_conv(self, net, copy, index, is_train):
        net = tf.layers.conv2d_transpose(
            net,
            filters=self['upconv.filters'] ** (index+1),
            kernel_size=2,
            strides=2,
            # regularizer?
            # kernel_regularizer=tf.contrib.layers.l2_regularizer(flags.reg),
            name="upsample_{}".format(name))

        return tf.concat([net, copy], axis=-1, name="concat_{}".format(index))

    def _scale(self, val):
        return (val / tf.constant(255.0)) * tf.constant(2.0) - tf.constant(1.0)

    def _unscale(self, val):
        return ((val + tf.constant(1.0)) / tf.constant(2.0)) * tf.constant(255.0)

    def configure(self, **kwargs):
        self.learning_rate = kwargs.get('learning_rate', 0.001)
        self.channels = kwargs.get('channels', 1)
        self.slice_width = kwargs.get('slice_width', 300)
        self.slice_height = kwargs.get('slice_height', 300)

    def build_graph(self):

        ###################
        # PLACEHOLDER
        ###################
        with tf.name_scope('placeholder'):
            x = log_1d(tf.placeholder(
                tf.float32, [None, self.slice_height, self.slice_width, self.channels], name="x"))
            x = self._scale(x)
            y = tf.placeholder(tf.float32, shape=x.shape, name="y")
            y = self._scale(y)
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
            output = log_1d(conv2d(net, self.channels,
                                   self['conv_n.kernel'], padding='same'))
            if self['conv_n.activation'] == 'tanh':
                output = tf.tanh(output)
            output = x - output
        ##################
        # PHASE IV: Loss & Optomizer
        #################
        loss = tf.nn.l2_loss(output - y)
        train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        # output = self._unscale(output)
        return dict(
            x=x,
            y=y,
            is_train=is_train,
            output=output,
            loss=loss,
            train_step=train_step
        )
