import tensorflow as tf
import operator
from tensorflow.contrib.layers import fully_connected
from tensorflow.python.eager import context
from tensorflow.python.util import nest
from tensorflow.contrib.rnn import RNNCell, LSTMStateTuple
from tensorflow.python.ops.rnn import _best_effort_input_batch_size
import math
import numpy as np


# Naive loop implementation where there is no parallelization
# and row-by-row execution
def naive_loop(cell, num_steps_x, num_steps_y, inputs_ta, parallel_iterations):

    one = tf.constant(1)

    # Function to get the sample skipping one row
    def get_up(t_, w_):
        return t_ - w_

    # Function to get the previous sample
    def get_last(t_, w_):
        return t_ - one

    total_steps = tf.constant(num_steps_x * num_steps_y)
    zero = tf.constant(0)
    num_steps_x = tf.constant(num_steps_x)
    num_steps_y = tf.constant(num_steps_y)
    # Body of the while loop operation that aplies the MD LSTM

    def loop(time_, outputs_ta_, states_ta_):

        # If the current position is less or equal than the width, we are in the first row
        # and we need to read the zero state we added in row (num_steps_x*num_steps_y).
        # If not, get the sample located at a width dstance.
        state_up = tf.cond(tf.less_equal(time_, num_steps_x),
                           lambda: states_ta_.read(total_steps),
                           lambda: states_ta_.read(get_up(time_, num_steps_x)))

        # If it is the first step we read the zero state if not we read the
        # inmediate last
        state_last = tf.cond(tf.less(zero, tf.mod(time_, num_steps_x)),
                             lambda: states_ta_.read(
            get_last(time_, num_steps_x)),
            lambda: states_ta_.read(total_steps))

        # We build the input state in both dimensions
        current_state = state_up[0], state_last[
            0], state_up[1], state_last[1]
        # Now we calculate the output state and the cell output
        out, state = cell(inputs_ta.read(time_), current_state)
        # We write the output to the output tensor array
        outputs_ta_ = outputs_ta_.write(time_, out)
        # And save the output state to the state tensor array
        states_ta_ = states_ta_.write(time_, state)

        # Return outputs and incremented time step
        return time_ + 1, outputs_ta_, states_ta_

    # Loop output condition. The index, given by the time, should be less than the
    # total number of steps defined within the image
    def condition(time_, outputs_ta_, states_ta_):
        return tf.less(time_, total_steps)
    return condition, loop

# Diagional parallelization


def diagonal_loop(cell, num_steps_x, num_steps_y, inputs_ta, parallel_iterations):
    max_d_step = tf.constant(num_steps_x + num_steps_y - 1)

    tf_num_steps_x = tf.constant(num_steps_x)
    tf_num_steps_y = tf.constant(num_steps_y)
    zero = tf.constant(0)
    one = tf.constant(1)
    total_steps = tf_num_steps_x * tf_num_steps_y

    def calc_pos(x, y):
        return y * tf_num_steps_x + x

    def loop(d_step, outputs_ta, states_ta):

        max_t_step = tf.cond(tf.less(d_step, tf_num_steps_y),
                             lambda: d_step + one, lambda: tf_num_steps_y)
        t_shift = tf.cond(tf.less(d_step, tf_num_steps_x),
                          lambda: zero, lambda: d_step - tf_num_steps_x + one)

        def inner_condition(t_step, outputs_ta_, states_ta_):
            return tf.less(t_step, max_t_step)

        def inner_loop(t_step, outputs_ta_, states_ta_):
            pos_x = d_step - t_step
            pos_y = t_step
            vert_pos = calc_pos(pos_x, pos_y)
            # If the current position is less or equal than the width, we are in the first row
            # and we need to read the zero state we added in row (num_steps_x*num_steps_y).
            # If not, get the sample located at a width dstance.
            state_top = tf.cond(tf.equal(pos_y, zero),
                                lambda: states_ta_.read(total_steps),
                                lambda: states_ta_.read(calc_pos(pos_x, pos_y - one)))

            # If it is the first step we read the zero state if not we read the
            # inmediate last
            state_left = tf.cond(tf.equal(pos_x, zero), lambda: states_ta_.read(total_steps),
                                 lambda: states_ta_.read(calc_pos(pos_x - one, pos_y)))

            # We build the input state in both dimensions
            current_state = state_top[0], state_left[
                0], state_top[1], state_left[1]

            # Now we calculate the output state and the cell output
            out, state = cell(inputs_ta.read(vert_pos), current_state)
            # We write the output to the output tensor array
            outputs_ta_ = outputs_ta_.write(vert_pos, out)
            # And save the output state to the state tensor array
            states_ta_ = states_ta_.write(vert_pos, state)

            return t_step + 1, outputs_ta_, states_ta_
        # Run the looped operation
        _, outputs_ta_, states_ta_ = tf.while_loop(inner_condition, inner_loop, [t_shift, outputs_ta, states_ta],
                                                   parallel_iterations=parallel_iterations)

        # Return outputs and incremented time step
        return d_step + 1, outputs_ta_, states_ta_

    # Here, we stop execution after
    def condition(d_step, outputs_ta_, states_ta_):
        return tf.less(d_step, max_d_step)

    return condition, loop

# Based on dynamic_rnn in https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/python/ops/rnn.py
# as well as
# https://github.com/philipperemy/tensorflow-multi-dimensional-lstm/blob/master/md_lstm.py


def rnn2d(cell, inputs, sequence_shape=(2, 2), initial_state=None,
          loop_implementation=naive_loop,
          dtype=None, parallel_iterations=None, scope=None, reverse_dims=None):
    """Creates a 2D recurrent neural network specified by 2D RNN `cell`.
    Performs fully dynamic unrolling of `inputs`.

    ```
    Args:
      cell: An instance of TwoDimensionalLSTMCell.
      inputs: The RNN inputs of shape [batch_size, width, height, ... further channels]
      sequence_shape: (optional) An int32/int64 tupel sized `[2]`.
      initial_state: (optional) An initial state for the RNN.
        If `cell.state_size` is an integer, this must be
        a `Tensor` of appropriate type and shape `[batch_size, cell.state_size]`.
        If `cell.state_size` is a tuple, this should be a tuple of
        tensors having shapes `[batch_size, s] for s in cell.state_size`.
      dtype: (optional) The data type for the initial state and expected output.
        Required if initial_state is not provided or RNN state has a heterogeneous
        dtype.
      parallel_iterations: (Default: 1).  The number of iterations to run in
        parallel.  Those operations which do not have any temporal dependency
        and can be run in parallel, will be.  This parameter trades off
        time for space.  Values >> 1 use more memory but take less time,
        while smaller values use less memory but computations take longer.
      scope: VariableScope for the created subgraph; defaults to "2d-rnn".
    Returns:
      A pair (outputs, state) where:
      outputs: The RNN output `Tensor`.
      state: The final state.  If `cell.state_size` is an int, this
        will be shaped `[batch_size, cell.state_size]`.  If it is a
        `TensorShape`, this will be shaped `[batch_size] + cell.state_size`.
        If it is a (possibly nested) tuple of ints or `TensorShape`, this will
        be a tuple having the corresponding shapes. If cells are `LSTMCells`
        `state` will be a tuple containing a `LSTMStateTuple` for each cell.
    Raises:
      TypeError: If `cell` is not an instance of RNNCell.
      ValueError: If inputs is None or an empty list.
    """

    # TODO: Assert cell type
    with tf.variable_scope(scope or "2d-rnn") as varscope:

        parallel_iterations = parallel_iterations or 1

        batch_size = inputs.get_shape().as_list()[0]
        full_reverse_dims = None

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

        # Reshape input data to a tensor containing the step indexes and features inputs
        # The batch size is inferred from the tensor size
        x = tf.reshape(
            inputs, [batch_size, num_steps_x, num_steps_y, features])

        # Reverse the selected dimensions
        if reverse_dims is not None:
            assert len(reverse_dims) is 2
            # We will not reverse the batch size and the channel
            full_reverse_dims = [False]
            full_reverse_dims.extend(reverse_dims)
            full_reverse_dims.append(False)
            reverse_dense = np.where(full_reverse_dims)[0]
            if len(reverse_dense) > 0:
                x = tf.reverse(x, reverse_dense)

         # Reorder inputs to (num_steps_x, num_steps_y, batch_size, features)
        x = tf.transpose(x, [1, 2, 0, 3])
        # Reshape to a one dimensional tensor of (num_steps_x * num_steps_y *
        # batch_size , features)
        x = tf.reshape(x, [-1, features])
        # Split tensor into h*w tensors of size (batch_size , features)
        x = tf.split(axis=0, num_or_size_splits=total_steps, value=x)

        # Create an input tensor array (literally an array of tensors) to use
        # inside the loop
        inputs_ta = tf.TensorArray(
            dtype=tf.float32, size=total_steps, name='input_ta', colocate_with_first_write_call=False)
        # Unestack the input X in the tensor array
        inputs_ta = inputs_ta.unstack(x)
        # Create an input tensor array for the states
        states_ta = tf.TensorArray(
            dtype=tf.float32, size=total_steps + 1, name='state_ta', clear_after_read=False, colocate_with_first_write_call=False)
        # And an other for the output
        outputs_ta = tf.TensorArray(
            dtype=tf.float32, size=total_steps, name='output_ta', colocate_with_first_write_call=False)

        states_ta = states_ta.write(total_steps, state)

        # Controls the initial index
        time = tf.constant(0)

        condition, loop = loop_implementation(
            cell, num_steps_x, num_steps_y, inputs_ta, parallel_iterations)

        # Run the looped operation
        _, outputs_ta, states_ta = tf.while_loop(condition, loop, [time, outputs_ta, states_ta],
                                                 parallel_iterations=1)

        # Extract the output tensors from the processesd tensor array
        outputs = outputs_ta.stack()
        states = states_ta.stack()

        # Reshape outputs to match the shape of the imput
        y = tf.reshape(
            outputs, [num_steps_x, num_steps_y, batch_size, outputs.shape[2]])

        # Reorder te dimensions to match the input
        y = tf.transpose(y, [2, 0, 1, 3])
        # Reverse if selected
        if reverse_dense is not None and len(reverse_dense) > 0:
            y = tf.reverse(y, reverse_dense)

        # Return the output and the inner states
        return y, states


def _patch_padding(inputs, sequence_length, axis=0):
    shape = inputs.get_shape().as_list()
    missing_padding = shape[axis] % sequence_length
    if missing_padding != 0:
        shape[axis] = sequence_length - missing_padding
        offset = tf.zeros(shape)
        inputs = tf.concat(axis=axis, values=[inputs, offset])
    return inputs


def prod(iterable):
    return reduce(operator.mul, iterable, 1)


def multidir_rnn2d(cell, inputs, sequence_shape=(2, 2), initial_state=None,
                   loop_implementation=naive_loop,
                   dtype=None, parallel_iterations=None, scope=None, reverse_dims=None):

    def build_rnn(rev_dims):
        return rnn2d(cell, inputs, sequence_shape=sequence_shape, initial_state=initial_state, loop_implementation=loop_implementation,
                     dtype=dtype, parallel_iterations=parallel_iterations, scope=scope, reverse_dims=rev_dims)

    rnn_lt, _ = build_rnn([False, False])
    rnn_lb, _ = build_rnn([False, True])
    rnn_rt, _ = build_rnn([True, False])
    rnn_rb, _ = build_rnn([True, True])

    return rnn_lt, rnn_lb, rnn_rt, rnn_rb


def multidir_conv(inputs, kernel_size=None, filters=None, strides=None, activation=tf.tanh, padding="valid"):
    def build_conv(x):
        return tf.layers.conv2d(inputs=x, strides=strides, filters=filters, kernel_size=kernel_size, padding=padding, activation=activation)
    return [build_conv(inputs[i]) for i in range(len(inputs))]


def multidir_fullyconnected(inputs, units=None, activation=tf.tanh):
    def build_fc(x):
        return fully_connected(inputs=x, num_outputs=units, activation_fn=activation)
    return [build_fc(inputs[i]) for i in range(len(inputs))]


def element_sum(inputs, reducer=None, axis=0):
    reducer = reducer or tf.reduce_sum
    return reducer(tf.stack(inputs, axis=0), axis=axis)


def sum_and_tanh(inputs, reducer=None):
    return tf.tanh(element_sum(inputs, reducer))
