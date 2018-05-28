import numpy as np
import tensorflow as tf


def wrap_4d(cell):
    print "-" * 70
    print "{:40}{:20} x 4".format(cell[0].name, cell[0].shape)
    return cell


def wrap_1d(cell):
    print "-" * 70
    print "{:40}{:20}".format(cell.name, cell.shape)
    return cell


def make_sparse(var):
    idx = tf.where(tf.not_equal(var, 0))
    return tf.SparseTensor(idx, tf.gather_nd(var, idx), var.get_shape())


def denseNDArrayToSparseTensor(arr, sparse_val=-1):
    idx = np.where(arr != sparse_val)
    return tf.SparseTensorValue(np.vstack(idx).T, arr[idx], arr.shape)
