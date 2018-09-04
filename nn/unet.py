import tensorflow as tf
from nn.layer.algorithmBaseV2 import AlgorithmBaseV2
from nn.util import log_1d
from nn.layer.histogrammed import conv2d, batch_normalization


DEFAULTS = {
    "depth": 5,
    "downconv": {
        "prepool": [False, True, True, True, True],
        "filters": 8,
    },
    "dropout": {
        "active": True,
        "prob": 0.5
    },
    "upconv": {
        "filters": 8,
        "activation": "",
    },
    "final": {
        "activation": "",
        "padding": "same"

    },
    "conv": {
        "activation": "relu",
        "padding": "same"
    }
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
            net = self.side_conv(
                net, self['downconv.filters'] * 2 ** (index), is_train)
        return net

    def pool(self, net, index):
        return log_1d(tf.layers.max_pooling2d(net, (2, 2), strides=(
            2, 2), name="pool_{}".format(index)))

    def side_conv(self, net, filters, is_train, dropout=False):
        for num in range(2):
            if dropout and self['dropout.active']:
                net = log_1d(tf.layers.dropout(
                    net, self['dropout.prob'], training=is_train, name='dropout'))
            with tf.variable_scope("conv_{}".format(num)):
                net = conv2d(net, filters, kernel_size=(
                    3, 3), padding=self['conv.padding'])
                net = batch_normalization(net, training=is_train)
                net = log_1d(activation(net, self['conv.activation']))
        return net

    def up_conv(self, net, copy, index, is_train):
        with tf.variable_scope("up_{}".format(index)):
            net = self.side_conv(
                net, self['upconv.filters'] * 2 ** (index), is_train)
            net = tf.layers.conv2d_transpose(
                net,
                filters=self['upconv.filters'] * 2 ** (index-1),
                kernel_size=2,
                strides=2,
                padding=self['conv.padding'],
                name="upsample_{}".format(index))
            net = log_1d(activation(net, self['upconv.activation']))
            return tf.concat([net, copy], axis=-1, name="concat_{}".format(index))

    def _scale(self, val):
        return (val / tf.constant(255.0)) * tf.constant(2.0) - tf.constant(1.0)

    def _unscale(self, val):
        return ((val + tf.constant(1.0)) / tf.constant(2.0)) * tf.constant(255.0)

    def configure(self, **kwargs):
        self.learning_rate = kwargs.get('learning_rate', 0.001)
        self.channels = kwargs.get('channels', 1)
        self.slice_width = kwargs.get('slice_width', 320)
        self.slice_height = kwargs.get('slice_height', 320)

    def network(self, input_, is_train):
        convs = []
        net = input_

        with tf.name_scope('down'):
            for idx in range(self['depth']):
                net = self.down_conv(
                    net, idx, self["downconv.prepool"][idx], is_train, dropout=(idx != 0))
                convs.append(net)
                net = self.pool(net, idx)

        with tf.name_scope('up'):
            for idx in reversed(range(self['depth'])):
                net = log_1d(self.up_conv(
                    net, convs[idx], idx+1, is_train, dropout=True))

        net = self.side_conv(net, self['upconv.filters'], is_train, True)
        net = conv2d(net, self.n_class, (1, 1), name='final',
                     activation=None, padding=self['final.padding'])
        return log_1d(activation(net, self['final.activation']))

    def build_graph(self):

        ###################
        # PLACEHOLDER
        ###################
        with tf.name_scope('placeholder'):
            x = log_1d(tf.placeholder(
                tf.float32, [None, self.slice_height, self.slice_width, self.channels], name="x"))
            # x = self._scale(x)
            y = tf.placeholder(tf.float32, shape=[
                               None, self.slice_height, self.slice_width, self.channels], name="y")
            # y = self._scale(y)
            _y = tf.cast(tf.reshape(y/tf.constant(255.0),
                                    [-1, self.slice_height, self.slice_width]), tf.int32)
            _y = log_1d(tf.one_hot(_y, self.n_class))
            is_train = tf.placeholder_with_default(False, (), name='is_train')

        output = self.network(x, is_train)

        ##################
        # Loss & Optomizer
        #################
        flat_logits = tf.reshape(output, [-1, self.n_class])
        flat_labels = tf.reshape(_y, [-1, self.n_class])
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
