import tensorflow as tf
from nn.layer.Layer import Layer
from nn.util import log_1d

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
        "train": False,
        "before_activation": False,
        "fused": True
    }
}


class CNNEncoder(Layer):

    def __init__(self, config, defaults=None, data_format='nhwc'):
        super(CNNEncoder, self).__init__(
            config, defaults or DEFAULTS, data_format)

    def _conv_block(self, net, index, is_train):
        if self['dropout.active'] and index > self['dropout.first_layer']-1:
            net = log_1d(tf.layers.dropout(
                net, self['dropout.prob'], training=is_train))

        _activation_fn = tf.nn.leaky_relu  # not yet configurable
        data_format = self._parse_format()

        def conv_layer(x):
            num_filters = (index + 1) * self['size']
            strides = self['strides']
            kernel = self['kernel']
            activation = _activation_fn if not self['bnorm.before_activation'] else None
            use_bias = self['bias']
            return log_1d(tf.layers.conv2d(x, num_filters, kernel, data_format=data_format, strides=strides, activation=activation, use_bias=use_bias))

        def batch_norm(x):
            is_training = is_train if self['bnorm.train'] else False
            axis = 1 if self._format == 'nchw' else 3
            return log_1d(tf.layers.batch_normalization(x, training=is_training, fused=self['bnorm.fused'], axis=axis))

        def pooling(x):
            pooling_sizes = self['pooling']
            _pool_data_format = data_format
            if index < len(pooling_sizes):
                pooling_size = pooling_sizes[index]
                x = log_1d(tf.layers.max_pooling2d(
                    x, pooling_size, pooling_size, data_format=_pool_data_format))
            return x

        net = conv_layer(net)
        if self['bnorm.active']:
            net = batch_norm(net)

        if self['bnorm.before_activation']:
            net = log_1d(_activation_fn(net))
            self.viz.append(net)

        net = pooling(net)

        return net

    def __call__(self, x, is_train):
        for i in range(self['layers']):
            x = self._conv_block(x, i, is_train)
        return x
