import numpy as np
import tensorflow as tf


def evaluate_device(gpuNumber):
    return "/device:CPU:0" if gpuNumber == -1 else "/device:GPU:{}".format(gpuNumber)


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


def getAlgorithm(name):
    from puigcerver2017 import Puigcerver2017
    from Voigtlaender2016 import VoigtlaenderDoetschNey2016
    from graves2009 import GravesSchmidhuber2009
    if name == "puigcerver":
        return Puigcerver2017()
    elif name == "voigtlaender":
        return VoigtlaenderDoetschNey2016()
    elif name == "graves":
        return GravesSchmidhuber2009()


def compare_outputs(dataset, pred, actual):
    pred = dataset.decompile(pred)
    actual = dataset.decompile(actual)
    out = '{:' + str(dataset.max_length) + '}  {}'
    return out.format(pred, actual)


def valueOr(config, prop, default):
    return config[prop] if prop in config and config['prop'] is not None else default
