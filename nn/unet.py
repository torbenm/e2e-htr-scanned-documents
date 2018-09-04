import tensorflow as tf
from nn.layer.algorithmBaseV2 import AlgorithmBaseV2
from nn.util import log_1d
from nn.layer.histogrammed import conv2d, batch_normalization


DEFAULTS = {
    "depth": 5,
    "downconv": {
        "prepool": [False, True, True, True, True],
        "filters": [8, 8]
        "activation": "relu",
        "padding": "same"
    },
    "upconv": {
        "filters": 8,
        "activation": "",
        "padding": "same"

    },
    "final": {
        "activation": "",
        "padding": "same"

    }
    "regularizer": 0.1
}


def activation(x, name):
    if name == "":
        return x
    elif name == "sigmoid":
        return tf.nn.signmoid(x)
    elif name == "relu":
        return tf.nn.relu(x)


class Unet(AlgorithmBaseV2):

    def __init__(self, config):
        self.n_class = 2
        super(Unet, self).__init__(config, DEFAULTS)

    def down_conv(self, net, index, prepool, is_train):
        with tf.variable_scope("down_{}".format(index)):
            if prepool:
                net = tf.layers.max_pooling2d(net, (2, 2), strides=(
                    2, 2), name="prepool_{}".format(index))
            for i, f in enumerate(self['downconv.filters']):
                # regularizer?
                net = conv2d(net, filters ** (index+1),
                             activation=None,
                             padding=self['downconv.padding'],
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                 self['regularizer']),
                             name="conv_{}".format(i))
                net = batch_normalization(net, training=train)
                net = activation(net, self['downconv.activation'])
                net = tf.nn.relu(net)
        return net

    def up_conv(self, net, copy, index, is_train):
        net = tf.layers.conv2d_transpose(
            net,
            filters=self['upconv.filters'] ** (index+1),
            kernel_size=2,
            strides=2,
            padding=self['upconv.padding'],
            # regularizer?
            kernel_regularizer=tf.contrib.layers.l2_regularizer(
                self['regularizer']),
            name="upsample_{}".format(name))
        net = activation(net, self['upconv.activation'])
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

    def network(self, input_, is_train):
        convs = []
        net = input_

        with tf.name_scope('down'):
            for idx in range(self['depth']):
                net = log_1d(self.downconv(
                    net, idx, self["downconv.prepool.{}".format(idx)], is_train))
                convs.append(net)

        with tf.name_scope('up'):
            for idx in reversed(range(self['depth'])):
                net = log_1d(self.upconv(net, convs[idx], idx, is_train))

        net = conv2d(conv9, 1, (1, 1), name='final',
                     activation=None, padding=self['final.padding'])
        return activation(net, self['final.activation'])

    def build_graph(self):

        ###################
        # PLACEHOLDER
        ###################
        with tf.name_scope('placeholder'):
            x = log_1d(tf.placeholder(
                tf.float32, [None, self.slice_height, self.slice_width, self.channels], name="x"))
            x = self._scale(x)
            y = tf.placeholder(tf.float32, shape=[
                               None, self.slice_height, self.slice_width, self.n_class], name="y")
            y = self._scale(y)
            is_train = tf.placeholder_with_default(False, (), name='is_train')

        output = self.network(x, is_train)

        ##################
        # Loss & Optomizer
        #################
        flat_logits = tf.reshape(output, [-1, self.n_class])
        flat_labels = tf.reshape(y, [-1, self.n_class])
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=flat_logits,
                                                                         labels=flat_labels))

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
