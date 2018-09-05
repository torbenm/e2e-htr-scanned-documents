import tensorflow as tf
import numpy as np
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


def pixel_wise_softmax(output_map):
    with tf.name_scope("pixel_wise_softmax"):
        max_axis = tf.reduce_max(output_map, axis=3, keepdims=True)
        exponential_map = tf.exp(output_map - max_axis)
        normalize = tf.reduce_sum(exponential_map, axis=3, keepdims=True)
        return exponential_map / normalize


class Unet(AlgorithmBaseV2):

    def __init__(self, config):
        self.n_class = 2
        super(Unet, self).__init__(config, DEFAULTS)

    def down_conv(self, net, index, prepool, is_train, dropout=False):
        with tf.variable_scope("down_{}".format(index)):
            net = self.side_conv(
                net, self['downconv.filters'] * 2 ** (index), is_train, dropout=dropout)
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

    def up_conv(self, net, copy, index, is_train, dropout=False):
        with tf.variable_scope("up_{}".format(index)):
            net = self.side_conv(
                net, self['upconv.filters'] * 2 ** (index), is_train, dropout=dropout)
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
        self.class_weights = kwargs.get('class_weights', None)

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

        if self.class_weights is not None:
            class_weights = tf.constant(
                np.array(self.class_weights, dtype=np.float32))

            weight_map = tf.multiply(flat_labels, class_weights)
            weight_map = tf.reduce_sum(weight_map, axis=1)

            loss_map = tf.nn.softmax_cross_entropy_with_logits_v2(logits=flat_logits,
                                                                  labels=flat_labels)
            weighted_loss = tf.multiply(loss_map, weight_map)

            loss = tf.reduce_mean(weighted_loss)
        else:
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=flat_logits,
                                                                             labels=flat_labels))

        train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        # output = self._unscale(output)
        output = pixel_wise_softmax(output)

        return dict(
            x=x,
            y=y,
            is_train=is_train,
            output=output,
            loss=loss,
            train_step=train_step
        )