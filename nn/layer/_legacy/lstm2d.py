import tensorflow as tf
from tensorflow.contrib.rnn import LayerRNNCell, LSTMStateTuple
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.layers import base as base_layer


_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"


# Adapted from https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/python/ops/rnn_cell_impl.py
# and
# https://github.com/philipperemy/tensorflow-multi-dimensional-lstm/blob/master/md_lstm.py


class LSTM2D(LayerRNNCell):
    """
    Two Dimensional LSTM recurrent network cell with peepholes.
    The implementation is based on: https://arxiv.org/pdf/0705.2011.pdf.
    We add forget_bias (default: 1) to the biases of the forget gate in order to
    reduce the scale of forgetting in the beginning of the training.
    It does not allow cell clipping, a projection layer, and does not
    use peep-hole connections: it is the basic baseline.

    Please be aware that state_is_tuple is always true.
    """

    def __init__(self, num_units, activation=None, use_peephole=False, gate_activation=None, reuse=None, name=None, return_tuple=True):
        """Initialize the multi dimensional LSTM cell.
        Args:
          num_units: int, The number of units in the LSTM cell.
          forget_bias: float, The bias added to forget gates (see above).
            Must set to `0.0` manually when restoring from CudnnLSTM-trained
            checkpoints.
          activation: Activation function of the inner states.  Default: `tanh`.
          gate_activation: Activation function of the gates.  Default: `sigmoid`.
          reuse: (optional) Python boolean describing whether to reuse variables
            in an existing scope.  If not `True`, and the existing scope already has
            the given variables, an error is raised.
          name: String, the name of the layer. Layers with the same name will
            share weights, but to avoid mistakes we require reuse=True in such
            cases.
          When restoring from CudnnLSTM-trained checkpoints, must use
          `CudnnCompatibleLSTMCell` instead.
        """
        super(LSTM2D,
              self).__init__(_reuse=reuse, name=name)
        self.input_spec = base_layer.InputSpec(ndim=2)
        self._num_units = num_units
        self.return_tuple = return_tuple
        self._use_peephole = use_peephole
        self._activation = activation or math_ops.tanh
        self._gate_activation = gate_activation or math_ops.sigmoid

    @property
    def state_size(self):
        return LSTMStateTuple(self._num_units, self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def build(self, inputs_shape):
        if inputs_shape[1].value is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                             % inputs_shape)

        input_depth = inputs_shape[1].value
        h_depth = self._num_units
        self._kernel = self.add_variable(
            _WEIGHTS_VARIABLE_NAME,
            shape=[input_depth + h_depth * 2, 5 * h_depth])
        self._bias = self.add_variable(
            _BIAS_VARIABLE_NAME,
            shape=[5 * h_depth],
            initializer=init_ops.zeros_initializer(dtype=self.dtype))
        self.built = True

    def call(self, inputs, state):
        """Long short-term memory cell (LSTM).
        Args:
          inputs: `2-D` tensor with shape `[batch_size, input_size]`.
          state: An `LSTMStateTuple` of state tensors, each shaped
            `[batch_size, self.state_size]`
        Returns:
          A pair containing the new hidden state, and the new state (`LSTMStateTuple`).
        """
        add = math_ops.add
        matmul = math_ops.matmul
        multiply = math_ops.multiply

        # one = constant_op.constant(1, dtype=tf.int32)
        # Parameters of gates are concatenated into one multiply for
        # efficiency.
        c1, c2, h1, h2 = state
        gate_inputs = matmul(
            array_ops.concat([inputs, h1, h2], 1), self._kernel)
        gate_inputs = nn_ops.bias_add(gate_inputs, self._bias)
        gate_inputs = self._gate_activation(gate_inputs)

        # i = input_gate, j = new_input, f1,f2 = forget_gates, o = output_gate
        i, j, f1, f2, o = array_ops.split(
            value=gate_inputs, num_or_size_splits=5, axis=1)

        # Here we use self._activation for j.
        # However, there might be performance considerations to apply the (same) activation
        # to all at once!
        # new_c = add(multiply(self._gate_activation(i), self._activation(j)),
        # add(multiply(c1, self._gate_activation(f1)), multiply(c2,
        # self._gate_activation(f2))))
        new_c = add(multiply(i, j),
                    add(multiply(c1, f1), multiply(c2, f2)))

        new_h = multiply(self._activation(new_c), self._gate_activation(o))

        new_state = LSTMStateTuple(
            new_c, new_h) if self.return_tuple else new_c
        return new_h, new_state
