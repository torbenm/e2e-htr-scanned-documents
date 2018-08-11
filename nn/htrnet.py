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
    "classifier": {
        "units": [512]
    },
    "dynamic_width": False,
    "format": "nhwc"
}


class HtrNet(AlgorithmBaseV2):

    def __init__(self, config):
        super(HtrNet, self).__init__(config, DEFAULTS)
        self.viz = []

    def _encoder(self, net, is_train):
        encoder = None
        if self['encoder.type'] == 'cnn':
            encoder = CNNEncoder.CNNEncoder(
                self['encoder.cnn'], data_format=self['format'])
        elif self['encoder.type'] == 'resnet':
            encoder = ResNetEncoder.ResNetEncoder(
                self['encoder.resnet'], data_format=self['format'])
        return encoder(net, is_train)

    def _classifier(self, net, is_train):
        axes = [1]  # [2, 3] if self['format'] == 'nchw' else [1, 2]
        net = log_1d(tf.reduce_mean(net, axes))
        for unit in self['classifier.units']:
            net = log_1d(tf.layers.dense(
                net, unit, activation=tf.nn.relu))
        net = log_1d(tf.layers.dense(net, 1, activation=None))
        return net

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
            self._config['format'] = DEFAULTS['format']

        ###################
        # PLACEHOLDER
        ###################
        with tf.name_scope('placeholder'):
            x = log_1d(tf.placeholder(
                tf.float32, [None, self.image_height, None if self['dynamic_width'] else self.image_width, self.channels], name="x"))
            y = tf.sparse_placeholder(
                tf.int32, shape=[None, None], name="y")
            class_y = tf.placeholder(
                tf.float32, shape=[None, 1], name="class_y")
            l = tf.placeholder(
                tf.int32, shape=[None], name="l")
            is_train = tf.placeholder_with_default(False, (), name='is_train')

            if self['format'] == 'nchw':
                net = log_1d(tf.transpose(x, [0, 3, 1, 2], name='nhwc2nchw'))
            else:
                net = x

        ################
        # PHASE I: Encoding
        ###############
        with tf.name_scope('encoder'):
            net = self._encoder(net, is_train)

            if self['format'] == 'nchw':
                net = log_1d(tf.transpose(
                    net, [0, 2, 3, 1], name='nchw2nhwc'))
            else:
                net = net

            net = log_1d(tf.transpose(net, [0, 2, 1, 3]))

        ################
        # PHASE II: Recurrent Block
        ###############
        with tf.name_scope('recurrent'):
            if self['dynamic_width']:
                # maybe theres a better way to do columnwise stacking
                net = log_1d(tf.reshape(
                    net, [-1, tf.shape(net)[1], net.shape[2] * net.shape[3]]))
            else:
                net = log_1d(tf.reshape(
                    net, [-1, net.shape[1], net.shape[2] * net.shape[3]]))
            encoder_net = net
            net = self._recurrent(net, is_train)

        ################
        # PHASE III: Fully Connected
        ###############
        with tf.name_scope('fc'):
            fc = FullyConnected.FullyConnected(self['fc'], self.vocab_length)
            net = fc(net, is_train)

        ##################
        # PHASE IV: CTC
        #################
        logits = log_1d(tf.transpose(net, [1, 0, 2]))

        with tf.name_scope('loss'):
            total_loss = tf.nn.ctc_loss(y, logits, l)
            tf.summary.scalar('loss', tf.reduce_mean(total_loss))

        with tf.name_scope('train'):
            train_step = self._train_step(total_loss, self.learning_rate)

        with tf.name_scope('logits'):
            logits = tf.nn.softmax(logits)

        #################
        # PHASE V: Classifier
        ################
        with tf.name_scope('classifier'):
            class_logits = self._classifier(encoder_net, is_train)
            class_pred = tf.nn.sigmoid(class_logits)
            class_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=class_logits, labels=class_y)
            # tf.summary.scalar('class_loss', tf.reduce_mean(class_loss))
            class_train = self._train_step(
                class_loss, self.class_learning_rate)

        """
        new_dict = {
            "classifier": {
                "x": x,
                "y": class_y,
                "train": class_train,
                "loss": class_loss,
                "logits": class_logits
            },
            "recognizer": {
                "x": x,
                "y": y,
                "l": l,
                "train": train_step,
                "loss": train_loss,
                "logits": logits
            }
            "is_train": is_train
        }
        """

        return dict(
            x=x,
            y=y,
            class_y=class_y,
            class_pred=class_pred,
            class_loss=class_loss,
            class_train=class_train,
            l=l,
            is_train=is_train,
            logits=logits,
            total_loss=total_loss,
            train_step=train_step,
            viz=self.viz
        )
