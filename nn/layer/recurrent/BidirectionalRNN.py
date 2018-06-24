from nn.layer.Layer import Layer
from nn.util import log_1d
import tensorflow as tf

DEFAULTS = {
    "layers": 5,
    "cell": "lstm",
    "units": 256,
    "dropout": 0.5
}


def _lstm_cell(units):
    cell = tf.nn.rnn_cell.LSTMCell(units)

    def summarizer():
        tf.summary.histogram('kernel', cell._kernel)
        tf.summary.histogram('bias', cell._bias)
    return cell, summarizer


def _gru_cell(units):
    return tf.nn.rnn_cell.GRUCell

    def summarizer():
        # currently none for gru
        # tf.summary.histogram('kernel', cell._kernel)
        # tf.summary.histogram('bias', cell._bias)
        pass
    return cell, summarizer


class BidirectionalRNN(Layer):

    def __init__(self, config, defaults=None, data_format='nhwc'):
        super(BidirectionalRNN, self).__init__(
            config, defaults or DEFAULTS, data_format)

    def _cell(self):
        cell_fn = _lstm_cell
        if self['cell'] == 'gru':
            cell_fn = _gru_cell
        return cell_fn(self['units'])

    def _rec_block(self, net, index, is_train, scope):
        with tf.variable_scope(scope):
            net = log_1d(tf.layers.dropout(
                net, self['dropout'], training=is_train, name='dropout'))
            with tf.name_scope('fw'):
                cell_fw, sum_fw = self._cell()
            with tf.name_scope('bw'):
                cell_bw, sum_bw = self._cell()

            output, _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw, net, dtype=tf.float32)
            net = log_1d(tf.concat(output, 2))
            sum_fw()
            sum_bw()
            return net

    def __call__(self, x, is_train):
        with tf.name_scope('brnn'):
            for i in range(self['layers']):
                x = self._rec_block(x, i, is_train, 'rnn{}'.format(i))
            return x
