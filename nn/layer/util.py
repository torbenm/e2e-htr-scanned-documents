import operator


def patch_padding(inputs, sequence_length, axis=0):
    shape = inputs.get_shape().as_list()
    missing_padding = shape[axis] % sequence_length
    if missing_padding != 0:
        shape[axis] = sequence_length - missing_padding
        offset = tf.zeros(shape)
        inputs = tf.concat(axis=axis, values=[inputs, offset])
    return inputs


def prod(iterable):
    return reduce(operator.mul, iterable, 1)
