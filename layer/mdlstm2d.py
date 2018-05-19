import tensorflow as tf
import operator
from tensorflow.contrib.layers import fully_connected
from tensorflow.python.eager import context
from tensorflow.python.util import nest
from tensorflow.contrib.rnn import RNNCell, LSTMStateTuple
from tensorflow.python.ops.rnn import _best_effort_input_batch_size
import math
import numpy as np

from lstm2d import LSTM2D
from rnn2d import _patch_padding

"""
Multidirectional 2D LSTM
"""
TOP_RIGHT = "TR"
TOP_LEFT = "TL"
BOTTOM_RIGHT = "BR"
BOTTOM_LEFT = "BL"


def pos_to_time(x, y, w, h, direction=TOP_LEFT):
    if direction == TOP_LEFT:
        return x + w * y
    if direction == TOP_RIGHT:
        return (w - x - 1) + w * y
    if direction == BOTTOM_LEFT:
        return x + (h - y - 1) * w
    if direction == BOTTOM_RIGHT:
        return (w - x - 1) + (h - y - 1) * w
    return -1


def do_step(cell, step_num, inputs, states):
    pass


def mdlstm2d(input, units, activation=None, sequence_shape=(2, 2), scope=None):
    cell = LSTM2D(units, activation=activation)

    with tf.variable_scope(scope or "2d-rnn") as varscope:

        ########################
        #   PREPROCESSING
        ########################

        if initial_state is not None:
            state = initial_state
        else:
            if not dtype:
                raise ValueError(
                    "If there is no initial_state, you must give a dtype.")
            state = cell.zero_state(batch_size, dtype)

        # Add padding to X axis, if necessary
        inputs = _patch_padding(inputs, sequence_shape[0], 1)
        # Add padding to Y axis, if necessary
        inputs = _patch_padding(inputs, sequence_shape[1], 2)

        shape = inputs.get_shape().as_list()

        num_steps_x, num_steps_y = int(
            shape[1] / sequence_shape[0]), int(shape[2] / sequence_shape[1])

        total_steps = num_steps_x * num_steps_y

        # Get the number of features (total number of imput values per step)
        features = sequence_shape[0] * sequence_shape[1] * prod(shape[3:])

        x = tf.reshape(
            inputs, [batch_size, num_steps_x, num_steps_y, features])
        x = tf.transpose(x, [1, 2, 0, 3])

        x = tf.unstack(tf.reshape(x, [-1, batch_size, features]))

        # TENSOR ARRAYS
        states_ta = tf.TensorArray(
            dtype=tf.float32, size=total_steps + 1, name='state_ta', clear_after_read=False, colocate_with_first_write_call=False)
        # And another for the output
        outputs_ta = tf.TensorArray(
            dtype=tf.float32, size=total_steps, name='output_ta', colocate_with_first_write_call=False)

        states_ta = states_ta.write(total_steps, state)

        ##########
        # LOOP
        ##########

        for d_step in range(num_steps_x + num_steps_y - 1):
