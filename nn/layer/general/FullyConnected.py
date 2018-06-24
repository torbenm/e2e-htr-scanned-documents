import tensorflow as tf
from nn.layer.Layer import Layer
from nn.util import log_1d

DEFAULTS = {
    "dropout": 0.5,
    "use_activation": True
}


class FullyConnected(Layer):

    def __init__(self, config, vocab_length, defaults=None, data_format='nhwc'):
        super(FullyConnected, self).__init__(
            config, defaults or DEFAULTS, data_format)
        self.vocab_length = vocab_length

    def __call__(self, x, is_train):
        x = log_1d(tf.layers.dropout(
            x, self['dropout'], training=is_train, name='dropout'))
        x = log_1d(tf.layers.dense(
            x, self.vocab_length, activation=tf.nn.relu if self['use_activation'] else None, name='dense'))
        return x
