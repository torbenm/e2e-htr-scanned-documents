import tensorflow as tf
from nn.layer.algorithmBase import AlgorithmBase
from nn.util import wrap_1d, wrap_4d, make_sparse, valueOr
from config.config import Configuration

"""
Puigcerver, Joan. "Are Multidimensional Recurrent Layers Really Necessary for Handwritten Text Recognition?."
"""


class Puigcerver2017(AlgorithmBase):

    def __init__(self, config, transpose=True):
        self.config = Configuration(config)
        self._transpose = transpose
        self.viz = []
        self.defaults = {
            'conv.num': 5,
            'conv.size': 16,
            'conv.pooling': [[2, 2]] * 3,
            'conv.bias': True,
            'conv.kernel': [3, 3],
            'conv.strides': [1, 1],
            'conv.dropout.prob': 0.2,
            'conv.dropout.first_layer': 2,
            'conv.dropout.active': True,
            'lstm.num': 5,
            'lstm.size': 256,
            'lstm.cell': 'lstm',
            'bnorm.active': True,
            'bnorm.train': False,
            'bnorm.before_activation': False,
            'format': 'nhwc',
            'fc.use_activation': True,
            'optimizer': 'adam'
        }

    def _conv_block(self, net, index, is_train):

        if self._get('conv.dropout.active') and index > self._get('conv.dropout.first_layer')-1:
            net = wrap_1d(tf.layers.dropout(net, self._get(
                'conv.dropout.prob'), training=is_train))

        _activation_fn = tf.nn.leaky_relu
        data_format = 'channels_first' if self._get(
            'format') == 'nchw' else 'channels_last'

        def conv_layer(x):
            num_filters = (index + 1) * self._get('conv.size')
            strides = self._get('conv.strides')
            kernel = self._get('conv.kernel')
            activation = _activation_fn if not self._get(
                'bnorm.before_activation') else None
            use_bias = self._get('conv.bias')
            return wrap_1d(tf.layers.conv2d(x, num_filters, kernel, data_format=data_format, strides=strides, activation=activation, use_bias=use_bias))

        def batch_norm(x):
            is_training = is_train if self._get('bnorm.train') else False
            axis = 1 if self._get(
                'format') == 'nchw' else 3
            return wrap_1d(tf.layers.batch_normalization(x, training=is_training, fused=True, axis=axis))

        def pooling(x):
            pooling_sizes = self._get('conv.pooling')
            _pool_data_format = data_format
            if index < len(pooling_sizes):
                pooling_size = pooling_sizes[index]
                x = wrap_1d(tf.layers.max_pooling2d(
                    x, pooling_size, pooling_size, data_format=_pool_data_format))
            return x

        net = conv_layer(net)
        if self._get('bnorm.active'):
            net = batch_norm(net)

        if self._get('bnorm.before_activation'):
            net = wrap_1d(_activation_fn(net))
            self.viz.append(net)

        net = pooling(net)

        return net

    def _rec_block(self, net, index, is_train, scope):
        with tf.variable_scope(scope):
            net = wrap_1d(tf.layers.dropout(net, 0.5, training=is_train))
            cell_fw, cell_bw = None, None
            if self._get('lstm.cell') == 'lstm':
                cell_fw = tf.nn.rnn_cell.LSTMCell(self._get('lstm.size'))
                cell_bw = tf.nn.rnn_cell.LSTMCell(self._get('lstm.size'))
            elif self._get('lstm.cell') == 'gru':
                cell_fw = tf.nn.rnn_cell.GRUCell(self._get('lstm.size'))
                cell_bw = tf.nn.rnn_cell.GRUCell(self._get('lstm.size'))

            output, _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw, net, dtype=tf.float32)
            net = wrap_1d(tf.concat(output, 2))
            return net

    def _get(self, prop):
        return self.config.default(prop, self.defaults[prop])

    def configure(self, image_width=200, image_height=100, batch_size=32, channels=1, vocab_length=62, sequence_length=100, learning_rate=0.001, class_learning_rate=0.001):
        self.image_width = image_width
        self.image_height = image_height
        self.channels = channels
        self.vocab_length = vocab_length
        self.learning_rate = learning_rate
        self.class_learning_rate = class_learning_rate

    def build_graph(self):

        # cpu does not support nchw, so nhwc forcing
        if(self._cpu):
            self.config['format'] = self.defaults['format']

        if self._transpose:
            x = wrap_1d(tf.placeholder(
                tf.float32, [None, self.image_width, self.image_height, self.channels], name="x"))
        else:
            x = wrap_1d(tf.placeholder(
                tf.float32, [None, self.image_height, self.image_width, self.channels], name="x"))

        y = tf.sparse_placeholder(
            tf.int32, shape=[None, sequence_length], name="y")
        l = tf.placeholder(
            tf.int32, shape=[None], name="y")
        is_train = tf.placeholder_with_default(False, (), name='is_train')

        if self._get('format') == 'nchw':
            net = wrap_1d(tf.transpose(x, [0, 3, 1, 2]))
        else:
            net = x

        for i in range(self._get('conv.num')):
            net = self._conv_block(net, i, is_train)

        if self._get('format') == 'nchw':
            net = wrap_1d(tf.transpose(net, [0, 2, 3, 1]))

        if not self._transpose:
            net = wrap_1d(tf.transpose(net, [0, 2, 1, 3]))
        net = wrap_1d(tf.reshape(
            net, [-1, net.shape[1], net.shape[2] * net.shape[3]]))

        for i in range(self._get('lstm.num')):
            net = self._rec_block(net, i, is_train, 'lstm-{}'.format(i))

        net = wrap_1d(tf.layers.dropout(net, 0.5, training=is_train))

        if self._transpose:
            net = wrap_1d(tf.contrib.layers.fully_connected(
                net, self.vocab_length))
        else:
            net = wrap_1d(tf.layers.dense(
                net, self.vocab_length, activation=tf.nn.relu if self._get('fc.use_activation') else None))

        logits = wrap_1d(tf.transpose(net, [1, 0, 2]))
        total_loss = tf.nn.ctc_loss(y, logits, l)

        logits = tf.nn.softmax(logits)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_step = None
        with tf.control_dependencies(update_ops):
            if self._get('optimizer') == 'adam':
                train_step = tf.train.AdamOptimizer(
                    self.learning_rate).minimize(total_loss)
            elif self._get('optimizer') == 'rmsprop':
                train_step = tf.train.RMSPropOptimizer(
                    self.learning_rate).minimize(total_loss)

        return dict(
            x=x,
            y=y,
            l=l,
            is_train=is_train,
            logits=logits,
            total_loss=total_loss,
            train_step=train_step,
            viz=self.viz
        )
