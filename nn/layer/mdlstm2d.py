import tensorflow as tf
# import operator
# from tensorflow.contrib.layers import fully_connected
# from tensorflow.python.eager import context
# from tensorflow.python.util import nest
# from tensorflow.contrib.rnn import RNNCell, LSTMStateTuple
# from tensorflow.python.ops.rnn import _best_effort_input_batch_size
# import math
from tensorflow.python.util import nest
import numpy as np

from lstm2d import LSTM2D
from util import patch_padding, prod

TOP_RIGHT = 0
TOP_LEFT = 1
BOTTOM_RIGHT = 2
BOTTOM_LEFT = 3


def mdlstm2d(num_units, inputs, blocks=(1, 1), activation=None,
             dtype=None, reducer=tf.reduce_mean, initial_state=None, scope=None):
    cell = LSTM2D(num_units, activation=activation)
    dtype = dtype or inputs.dtype
    with tf.variable_scope(scope or "4d-lstm2d") as varscope:
        # ---------- INITIALIZE -----------
        shape = inputs.get_shape().as_list()
        batch_size = shape[0]
        print shape
        # ---------- CALCULATE STEPS ----------
        num_steps_x, num_steps_y = int(
            shape[1] / blocks[0]), int(shape[2] / blocks[1])
        total_steps = num_steps_x * num_steps_y
        features = blocks[0] * blocks[1] * prod(shape[3:])

        print prod(shape[3:]), num_units

        # ---------- PREPROCESS ----------
        state = _init_state(initial_state, cell,
                            dtype, batch_size)
        inputs = _apply_padding(inputs, blocks)
        x = tf.reshape(
            inputs, [batch_size, -1, features])
        x = tf.transpose(x, [1, 0, 2])
        x = tf.unstack(x)

        # ----------- TENSOR ARRAYS -----------
        inputs_ta = tf.TensorArray(
            dtype=tf.float32, size=total_steps, name='input_ta', colocate_with_first_write_call=False)
        # Unestack the input X in the tensor array
        inputs_ta = inputs_ta.unstack(x)
        states_ta = tf.TensorArray(
            dtype=dtype, size=total_steps + 1), name = 'state_ta', clear_after_read = False, colocate_with_first_write_call = True)
        outputs_ta=tf.TensorArray(
            dtype = dtype, size = total_steps, name = 'output_ta', colocate_with_first_write_call = True)

        states_ta=states_ta.write(total_steps, state)

        # ---------- LOOP ---------------
        _, outputs_ta, states_ta=tf.while_loop(condition, loop, [time, outputs_ta, states_ta],
                                                 parallel_iterations = 1)

        # ---------- POSTPROCESS --------
        outputs=outputs_ta.stack()

        # TODO: check correctness
        y=tf.reshape(
            outputs, [num_steps_x, num_steps_y, batch_size, features * num_units])
        y=tf.transpose(y, [2, 0, 1, 3])

    return y


def _loop_condition():
    pass


def _loop_body():
    pass


def _init_state(initial_state, cell, dtype, batch_size):
    if initial_state is not None:
        state=initial_state
    else:
        if not dtype:
            raise ValueError(
                "If there is no initial_state, you must give a dtype.")
        state=cell.zero_state(batch_size, dtype)
    return state


def _apply_padding(inputs, blocks):
    inputs=patch_padding(inputs, blocks[0], 1)
    return patch_padding(inputs, blocks[1], 2)


def _pos_to_time(x, y, w, h, direction = TOP_LEFT):
    if direction == TOP_LEFT:
        return x + w * y
    if direction == TOP_RIGHT:
        return (w - x - 1) + w * y
    if direction == BOTTOM_LEFT:
        return x + (h - y - 1) * w
    if direction == BOTTOM_RIGHT:
        return (w - x - 1) + (h - y - 1) * w
    return -1
