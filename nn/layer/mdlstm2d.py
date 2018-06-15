import tensorflow as tf
from tensorflow.python.util import nest
import numpy as np
from tensorflow.python.ops import array_ops
from tensorflow.contrib.rnn import LSTMStateTuple

from .lstm2d import LSTM2D
from .util import patch_padding, prod

TOP_LEFT = 0
TOP_RIGHT = 1
BOTTOM_LEFT = 2
BOTTOM_RIGHT = 3


def mdlstm2d(num_units, inputs, blocks=(1, 1), activation=None,
             dtype=None, reducer=tf.reduce_mean, initial_state=None, scope=None):
    cell = LSTM2D(num_units, activation=activation, return_tuple=False)
    dtype = dtype or inputs.dtype
    with tf.variable_scope(scope or "4d-lstm2d") as varscope:
        # ---------- INITIALIZE -----------
        shape = inputs.get_shape().as_list()
        batch_size = shape[0]
        time = tf.constant(0)
        one = tf.constant(1)
        zero = tf.constant(0)
        # ---------- CALCULATE STEPS ----------
        num_steps_x, num_steps_y = int(
            shape[1] / blocks[0]), int(shape[2] / blocks[1])
        total_steps = num_steps_x * num_steps_y
        features = blocks[0] * blocks[1] * prod(shape[3:])

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
            dtype=tf.float32, size=total_steps, name='input_ta', colocate_with_first_write_call=True, clear_after_read=False)
        # Unestack the input X in the tensor array
        inputs_ta = inputs_ta.unstack(x)
        states_ta = tf.TensorArray(
            dtype=dtype, size=(total_steps + 1) * 4, name='state_ta', clear_after_read=False, colocate_with_first_write_call=True)
        outputs_ta = tf.TensorArray(
            dtype=dtype, size=total_steps * 4, name='output_ta', colocate_with_first_write_call=True)

        states_ta = states_ta.write(total_steps, state)
        states_ta = states_ta.write(total_steps * 2 + 1, state)
        states_ta = states_ta.write(total_steps * 3 + 2, state)
        states_ta = states_ta.write(total_steps * 4 + 3, state)

        total_steps = tf.constant(total_steps)
        num_steps_x = tf.constant(num_steps_x)
        num_steps_y = tf.constant(num_steps_y)

        # ---------- LOOP ---------------

        def _translate_pos(time_, dir_, mul=True):
            pos = time_
            if dir_ == TOP_RIGHT:
                pos = ((tf.floordiv(time_, num_steps_x) + 1) * num_steps_x) - \
                    tf.floormod(time_, num_steps_x) - 1
            elif dir_ == BOTTOM_RIGHT:
                pos = (total_steps - 1) - time_
            elif dir_ == BOTTOM_LEFT:
                pos = total_steps - (tf.floordiv(time_, num_steps_x) + 1) * num_steps_x + \
                    tf.floormod(time_, num_steps_x)
            if mul:
                return (total_steps * dir_) + pos
            return pos

        def _get_fpos(dir_):
            return total_steps * dir_ + dir_

        def get_input_state_for_dir(time_, states_ta_, dir_):

            state_top = tf.cond(tf.less_equal(time_, num_steps_x),
                                lambda: states_ta.read(
                                    _get_fpos(dir_) + total_steps),
                                lambda: states_ta_.read(_get_fpos(dir_) + time_ - num_steps_x))

            state_left = tf.cond(tf.less(zero, tf.mod(time_, num_steps_x)),
                                 lambda: states_ta_.read(
                                     _get_fpos(dir_) + time_ - one),
                                 lambda: states_ta.read(_get_fpos(dir_) + total_steps))

            # We build the input state in both dimensions
            current_state = state_top[0], state_left[
                0], state_top[1], state_left[1]
            inputs = inputs_ta.read(_translate_pos(time_, dir_, False))
            return current_state, inputs

        def loop(time_, outputs_ta_, states_ta_):

            state_TL, inputs_TL = get_input_state_for_dir(
                time_, states_ta_, TOP_LEFT)
            state_TR, inputs_TR = get_input_state_for_dir(
                time_, states_ta_, TOP_RIGHT)
            state_BL, inputs_BL = get_input_state_for_dir(
                time_, states_ta_, BOTTOM_LEFT)
            state_BR, inputs_BR = get_input_state_for_dir(
                time_, states_ta_, BOTTOM_RIGHT)

            states_ = [state_TL, state_TR, state_BL, state_BR]
            states_ = [array_ops.concat(
                [state[i] for state in states_], axis=0) for i in range(4)]

            inputs_ = array_ops.concat(
                [inputs_TL, inputs_TR, inputs_BL, inputs_BR], axis=0)

            # Now we calculate the output state and the cell output
            outputs_, states_ = cell(inputs_, states_)

            outputs_TL, outputs_TR, outputs_BL, outputs_BR = array_ops.split(
                value=outputs_, num_or_size_splits=4, axis=0)
            states_TL, states_TR, states_BL, states_BR = array_ops.split(
                value=states_, num_or_size_splits=4, axis=0)

            # We write the output to the output tensor array
            outputs_ta_ = outputs_ta_.write(
                _translate_pos(time_, TOP_LEFT), outputs_TL)
            outputs_ta_ = outputs_ta_.write(
                _translate_pos(time_, TOP_RIGHT), outputs_TR)
            outputs_ta_ = outputs_ta_.write(
                _translate_pos(time_, BOTTOM_LEFT), outputs_BL)
            outputs_ta_ = outputs_ta_.write(
                _translate_pos(time_, BOTTOM_RIGHT), outputs_BR)

            # And save the output state to the state tensor array
            states_ta_ = states_ta_.write(
                _get_fpos(TOP_LEFT) + time_, LSTMStateTuple(outputs_TL, states_TL), name="states_tl")
            states_ta_ = states_ta_.write(
                _get_fpos(TOP_RIGHT) + time_, LSTMStateTuple(outputs_TR, states_TR), name="states_tr")
            states_ta_ = states_ta_.write(
                _get_fpos(BOTTOM_LEFT) + time_, LSTMStateTuple(outputs_BL, states_BL), name="states_bl")
            states_ta_ = states_ta_.write(
                _get_fpos(BOTTOM_RIGHT) + time_, LSTMStateTuple(outputs_BR, states_BR), name="states_br")

            # Return outputs and incremented time step
            return time_ + 1, outputs_ta_, states_ta_

        # Loop output condition. The index, given by the time, should be less than the
        # total number of steps defined within the image
        def condition(time_, outputs_ta_, states_ta_):
            return tf.less(time_, total_steps)

        _, outputs_ta, _ = tf.while_loop(condition, loop, [time, outputs_ta, states_ta],
                                         parallel_iterations=1)

        # ---------- POSTy
        outputs = outputs_ta.stack()

        y = tf.reshape(
            outputs, [4, num_steps_x, num_steps_y, batch_size, outputs.shape[2]])
        y = reducer(y, axis=0)
        y = tf.transpose(y, [2, 0, 1, 3])

    return y


def _init_state(initial_state, cell, dtype, batch_size):
    if initial_state is not None:
        state = initial_state
    else:
        if not dtype:
            raise ValueError(
                "If there is no initial_state, you must give a dtype.")
        state = cell.zero_state(batch_size, dtype)
    return state


def _apply_padding(inputs, blocks):
    inputs = patch_padding(inputs, blocks[0], 1)
    return patch_padding(inputs, blocks[1], 2)
