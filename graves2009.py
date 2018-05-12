import tensorflow as tf
from layer.lstm2d import LSTM2D
from layer.rnn2d import multidir_rnn2d, multidir_conv, sum_and_tanh, element_sum, multidir_fullyconnected
from layer.algorithmBase import AlgorithmBase
from util import wrap_1d, wrap_4d, make_sparse
# Taken from
# Graves, Alex, and Juergen Schmidhuber. "Offline handwriting recognition
# with multidimensional recurrent neural networks." Advances in neural
# information processing systems. 2009.


class GravesSchmidhuber2009(AlgorithmBase):

    def build_graph(self, image_width=200, image_height=100, batch_size=32, channels=1, vocab_length=62, sequence_length=100, learning_rate=0.0001):

        # 32 x 300 x 30 x 1
        x = wrap_1d(tf.placeholder(
            tf.float32, [batch_size, image_width, image_height, channels], name="x"))
        # 32 x 84
        y = tf.sparse_placeholder(
            tf.int32, shape=[batch_size, sequence_length], name="y")
        #y = tf.Variable(y)
        #y = make_sparse(y)

        # MDLSTM Layer 1
        lstm_cell = LSTM2D(2)
        # 4 x [32 x 150 x 15 x 2]
        net = wrap_4d(multidir_rnn2d(
            lstm_cell, x, sequence_shape=(1, 2), scope='lstm-1', dtype=tf.float32))
        # 4 x [32 x 75 x 3 x 6]
        net = wrap_4d(multidir_conv(net, kernel_size=(1, 2),
                                    strides=(1, 2), filters=6))
        # 32 x 75 x 3 x 6
        net = wrap_1d(sum_and_tanh(net))

        # MDLSTM layer 2
        lstm_cell = LSTM2D(10)
        # 4 x [32 x 38 x 1 x 10]
        net = wrap_4d(multidir_rnn2d(
            lstm_cell, net, sequence_shape=(1, 2), scope='lstm-2', dtype=tf.float32))  # is this really the sequence shape?
        net = wrap_4d(multidir_conv(net, kernel_size=(2, 2),
                                    strides=(2, 2), filters=20))
        net = wrap_1d(sum_and_tanh(net))

        # MDLSTM layer 3
        lstm_cell = LSTM2D(50)
        net = wrap_4d(multidir_rnn2d(
            lstm_cell, net, sequence_shape=(1, 2), scope='lstm-3', dtype=tf.float32))  # is this really the sequence shape?
        net = wrap_4d(multidir_fullyconnected(net, units=vocab_length))
        logits = wrap_1d(element_sum(net, axis=[0, 3]))
        # logits = tf.nn.softmax(net)
        logits = wrap_1d(tf.transpose(logits, [1, 0, 2]))
        decoded, _ = tf.nn.ctc_greedy_decoder(
            logits, tf.constant([sequence_length] * batch_size))
        # wrap_1d(decoded[0])

        total_loss = tf.nn.ctc_loss(
            y, logits, tf.constant([sequence_length] * batch_size))
        train_step = tf.train.AdamOptimizer(
            learning_rate).minimize(total_loss)

        return dict(
            x=x,
            y=y,
            output=decoded,
            total_loss=total_loss,
            train_step=train_step,
            saver=tf.train.Saver()
        )
