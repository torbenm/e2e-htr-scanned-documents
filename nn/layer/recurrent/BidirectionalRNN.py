from nn.layer.Layer import Layer
from nn.util import log_1d

DEFAULTS = {
    "layers": 5,
    "cell": "lstm",
    "units": 256,
    "dropout": 0.5
}


class BidirectionalRNN(Layer):

    def __init__(self, config, defaults=None, data_format='nhwc'):
        super(BidirectionalRNN, self).__init__(
            config, defaults or DEFAULTS, data_format)

    def _cell(self):
        cell_fn = tf.nn.rnn_cell.LSTMCell
        if self['cell'] == 'gru':
            cell_fn = tf.nn.rnn_cell.GRUCell
        return cell_fn(self['units'])

    def _rec_block(self, net, index, is_train, scope):
        with tf.variable_scope(scope):
            net = log_1d(tf.layers.dropout(
                net, self['dropout'], training=is_train))
            cell_fw = self._cell()
            cell_bw = self._cell()

            output, _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw, net, dtype=tf.float32)
            net = log_1d(tf.concat(output, 2))
            return net

    def __call__(self, x, is_train):
        for i in range(self['layers']):
            x = self._rec_block(x, i, is_train, 'rnn-{}'.format(i))
        return x
