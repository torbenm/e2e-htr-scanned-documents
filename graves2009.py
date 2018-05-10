import tensorflow as tf
from layer.lstm2d import LSTM2D
from layer.rnn2d import multidir_rnn2d, multidir_conv, sum_and_tanh, element_sum, multidir_fullyconnected

# Taken from
# Graves, Alex, and Jürgen Schmidhuber. "Offline handwriting recognition
# with multidimensional recurrent neural networks." Advances in neural
# information processing systems. 2009.


class GravesSchmidhuber2009(AlgorithmBase):

    def build_graph(self, batch_size=32, vocab_length):

        x = tf.placeholder(tf.float32, [batch_size, h, w, channels])
        y = tf.placeholder(tf.float32, [batch_size, h, w, channels])

        # MDLSTM Layer 1
        lstm_cell = LSTM2D(2)
        net = multidir_rnn2d(
            lstm_cell, x, sequence_shape=(2, 2), scope='lstm-1')
        net = multidir_conv(net, kernel_size=(2, 4, 2),
                            stride=(2, 4, 2), filters=6)
        net = sum_and_tanh(net)

        # MDLSTM layer 2
        lstm_cell = LSTM2D(10)
        net = multidir_rnn2d(
            lstm_cell, x, sequence_shape=(2, 4), scope='lstm-2')  # is this really the sequence shape?
        net = multidir_conv(net, kernel_size=(2, 4, 10),
                            stride=(2, 4, 10), filters=20)
        net = sum_and_tanh(net)

        # MDLSTM layer 3
        lstm_cell = LSTM2D(50)
        net = multidir_rnn2d(
            lstm_cell, x, sequence_shape=(2, 4), scope='lstm-3')  # is this really the sequence shape?
        net = multidir_fullyconnected(net, units=vocab_length)
        net = element_sum(net, axis=[0, 1])

        net = tf.nn.softmax(net)

        # next: ctc
        # sequence lengths values are missing
        net = tf.nn.ctc_greedy_decoder(net, ??)
        return net
