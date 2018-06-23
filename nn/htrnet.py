import tensorflow as tf
from nn.layer.algorithmBaseV2 import AlgorithmBaseV2
from nn.util import log_1d
from nn.layer.encoder import CNNEncoder, ResNetEncoder
from nn.layer.recurrent import BidirectionalRNN
from nn.layer.general import FullyConnected

DEFAULTS = {
    "encoder": {
        "type": "cnn",
        "cnn": {},
        "resnet": {}
    },
    "recurrent": {
        "type": "brnn",  # This is just a placeholder for now as only one type of recurrent block is supported
        "brnn": {}
    },
    "format": "nhwc"
}


class HtrNet(AlgorithmBaseV2):

    def __init__(self, config):
        super(HtrNet, self).__init__(config, DEFAULTS)
        self.viz = []

    def _encoder(self, net, is_train):
        encoder = None
        if self['encoder'] == 'cnn':
            encoder = CNNEncoder.CNNEncoder(
                self['encoder.cnn'], data_format=self['format'])
        elif self['encoeder'] == 'resnet':
            encoder = ResNetEncoder.ResNetEncoder(
                self['encoder.resnet'], data_format=self['format'])
        return encoder(net, is_train)

    def _recurrent(self, net, is_train):
        recurrent_block = BidirectionalRNN.BidirectionalRNN(
            self['recurrent.brnn'], data_format=self['format'])
        return recurrent_block(net, is_train)

    def _train_step(self, total_loss, learning_rate):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_step = None
        with tf.control_dependencies(update_ops):
            if self['optimizer'] == 'adam':
                train_step = tf.train.AdamOptimizer(
                    learning_rate).minimize(total_loss)
            elif self['optimizer'] == 'rmsprop':
                train_step = tf.train.RMSPropOptimizer(
                    learning_rate).minimize(total_loss)
        return train_step

    def build_graph(self, image_width=200, image_height=100, batch_size=32, channels=1, vocab_length=62, sequence_length=100, learning_rate=0.001):

        # cpu does not support nchw, so nhwc forcing
        if(self._cpu):
            self._config['format'] = DEFAULTS['format']

        ###################
        # PLACEHOLDER
        ###################

        x = log_1d(tf.placeholder(
            tf.float32, [None, image_height, image_width, channels], name="x"))

        y = tf.sparse_placeholder(
            tf.int32, shape=[None, sequence_length], name="y")
        l = tf.placeholder(
            tf.int32, shape=[None], name="y")
        is_train = tf.placeholder_with_default(False, (), name='is_train')

        if self['format'] == 'nchw':
            net = log_1d(tf.transpose(x, [0, 3, 1, 2]))
        else:
            net = x

        ################
        # PHASE I: Encoding
        ###############

        net = self._encoder(net, is_train)

        if self['format'] == 'nchw':
            net = log_1d(tf.transpose(net, [0, 2, 3, 1]))

        ################
        # PHASE II: Recurrent Block
        ###############

        net = log_1d(tf.reshape(
            net, [-1, net.shape[1], net.shape[2] * net.shape[3]]))

        net = self._recurrent(net, is_train)

        ################
        # PHASE III: Fully Connected
        ###############

        fc = FullyConnected.FullyConnected(self['fc'], vocab_length)
        net = fc(net, is_train)

        ##################
        # PHASE IV: CTC
        #################
        logits = log_1d(tf.transpose(net, [1, 0, 2]))
        total_loss = tf.nn.ctc_loss(y, logits, l)

        logits = tf.nn.softmax(logits)
        train_step = self._train_step(total_loss, learning_rate)

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
