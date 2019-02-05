import tensorflow as tf
from lib.nnlayer.Layer import Layer
from lib.nnutil import log_1d
from lib.nnlayer.histogrammed import conv2d, batch_normalization
from lib.nnlayer.general.group_norm import group_norm

DEFAULTS = {
    "layers": 5,
    "size": 16,
    "pooling": [[2, 2]] * 3,
    "bias": True,
    "kernel": [3, 3],
    "strides": [1, 1],
    "dropout": {
        "active": True,
        "first_layer": 2,
        "prob": 0.2
    },
    "bnorm": {
        "active": True,
        "fused": True
    },
    "groupnorm": False
}


class CNNEncoder(Layer):

    def __init__(self, config, defaults=None, data_format='nhwc'):
        super(CNNEncoder, self).__init__(
            config, defaults or DEFAULTS, data_format)
        self.viz = []

    def _conv_block(self, net, index, is_train):
        if self['dropout.active'] and index > self['dropout.first_layer']-1:
            net = log_1d(tf.layers.dropout(
                net, self['dropout.prob'], training=is_train, name='dropout'))

        _activation_fn = tf.nn.leaky_relu  # not yet configurable
        data_format = self._parse_format()

        def conv_layer(x):
            with tf.name_scope('conv'):
                num_filters = (index + 1) * self['size']
                return log_1d(conv2d(x, num_filters, self['kernel'], data_format=data_format,
                                     strides=self['strides'], activation=None, use_bias=self['bias']))

        def batch_norm(x):
            axis = 1 if self._format == 'nchw' else 3
            return log_1d(batch_normalization(x, training=is_train, fused=self['bnorm.fused'], axis=axis, name='batch_norm'))

        def groupnorm(x):
            return group_norm(x, self['groupnorm']) if self['groupnorm'] else x

        def pooling(x):
            pooling_sizes = self['pooling']
            _pool_data_format = data_format
            if index < len(pooling_sizes):
                pooling_size = pooling_sizes[index]
                x = log_1d(tf.layers.max_pooling2d(
                    x, pooling_size, pooling_size, data_format=_pool_data_format, name='maxpool'))
            return x

        net = conv_layer(net)
        if self['bnorm.active']:
            with tf.name_scope('bnorm'):
                net = batch_norm(net)

        net = log_1d(_activation_fn(net))
        tf.summary.histogram('final_activation', net)
        self.viz.append(net)

        net = pooling(net)

        return net

    def __call__(self, x, is_train):
        with tf.name_scope('convolutions'):
            for i in range(self['layers']):
                with tf.variable_scope('conv{}'.format(i)):
                    x = self._conv_block(x, i, is_train)
        return x
