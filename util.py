import numpy as np
import tensorflow as tf


def evaluate_device(gpuNumber):
    return "/device:CPU:0" if gpuNumber == -1 else "/device:GPU:{}".format(gpuNumber)


def make_sparse(var):
    idx = tf.where(tf.not_equal(var, 0))
    return tf.SparseTensor(idx, tf.gather_nd(var, idx), var.get_shape())


def denseNDArrayToSparseTensor(arr, sparse_val=-1):
    idx = np.where(arr != sparse_val)
    return tf.SparseTensorValue(np.vstack(idx).T, arr[idx], arr.shape)


def compare_outputs(dataset, pred, actual):
    pred = dataset.decompile(pred)
    actual = dataset.decompile(actual)
    out = '{:' + str(dataset.max_length) + '}  {}'
    return out.format(pred, actual)
