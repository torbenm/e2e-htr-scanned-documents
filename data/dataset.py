from . import util
import os
import math
import numpy as np
import cv2


class Dataset(object):

    def __init__(self, name, transpose=True):
        self.name = name
        self.datapath = os.path.join(util.OUTPUT_PATH, name)
        self._load_vocab()
        self._load_meta()
        self._load_sets()
        self._calc_max_length()
        self._compile_sets()
        self.transpose = transpose
        self.channels = 1

    def _load_meta(self):
        self.meta = util.loadJson(self.datapath, "meta")

    def _load_vocab(self):
        self.vocab = util.loadJson(self.datapath, "vocab")
        self.vocab_length = len(self.vocab[0])

    def _load_sets(self):
        self.data = {
            "train": util.loadJson(self.datapath, "train"),
            "dev": util.loadJson(self.datapath, "dev"),
            "test": util.loadJson(self.datapath, "test")
        }

    def _compile_set(self, dataset):
        for item in self.data[dataset]:
            item['compiled'] = self.compile(item['truth'])

    def _compile_sets(self):
        self._compile_set("train")
        self._compile_set("dev")
        self._compile_set("test")

    def _calc_max_length(self):
        _all = []
        _all.extend(self.data["train"])
        _all.extend(self.data["test"])
        _all.extend(self.data["dev"])
        self.max_length = max(map(lambda x: len(x["truth"]), _all))

    def compile(self, text):
        parsed = [self.vocab[1][c] for c in text]
        parsed.extend([-1] * (self.max_length - len(text)))
        return parsed

    def decompile(self, values):
        def getKey(key):
            try:
                return self.vocab[0][str(key)]
            except KeyError:
                return ''
        return ''.join([getKey(c) for c in values])

    def _loadline(self, line, transpose=True):
        l = len(line["compiled"])
        y = np.asarray(line["compiled"])
        x = cv2.imread(line["path"], cv2.IMREAD_GRAYSCALE)
        if transpose:
            try:
                x = np.transpose(x, [1, 0])
            except ValueError:
                return None, None, None, None
            if x.shape[0] != self.meta["width"] or x.shape[1] != self.meta["height"]:
                x = pad(x, (self.meta["width"], self.meta["height"]))
            x = np.reshape(x, [self.meta["width"], self.meta["height"], 1])
        else:
            if x.shape[1] != self.meta["width"] or x.shape[0] != self.meta["height"]:
                x = pad(x, (self.meta["height"], self.meta["width"]))
            x = np.reshape(x, [self.meta["height"], self.meta["width"], 1])
        return x, y, l, line["path"]

    def _load_batch(self, index, batch_size, dataset, with_filepath=False):
        X = []
        Y = []
        L = []
        F = []
        for idx in range(index * batch_size, min((index + 1) * batch_size, len(self.data[dataset]))):
            x, y, l, f = self._loadline(
                self.data[dataset][idx], self.transpose)
            if x is not None:
                X.append(x)
                Y.append(y)
                L.append(l)
                F.append(f)
        if not with_filepath:
            return np.asarray(X), np.asarray(Y), np.asarray(L)
        else:
            return np.asarray(X), np.asarray(Y), np.asarray(L), F

    def generateBatch(self, batch_size, max_batches=0, dataset="train", with_filepath=False):
        num_batches = self.getBatchCount(batch_size, max_batches, dataset)
        for b in range(num_batches):
            yield self._load_batch(b, batch_size, dataset, with_filepath)
        pass

    def generateEpochs(self, batch_size, num_epochs, max_batches=0, dataset="train", with_filepath=False):
        for e in range(num_epochs):
            yield self.generateBatch(batch_size, max_batches=max_batches, dataset=dataset, with_filepath=with_filepath)

    def getBatchCount(self, batch_size, max_batches=0, dataset="train"):
        total_len = len(self.data[dataset])
        num_batches = int(math.ceil(float(total_len) / batch_size))
        return min(
            num_batches, max_batches) if max_batches > 0 else num_batches


def pad(array, reference_shape, offsets=None):
    """
    array: Array to be padded
    reference_shape: tuple of size of ndarray to create
    offsets: list of offsets (number of elements must be equal to the dimension of the array)
    will throw a ValueError if offsets is too big and the reference_shape cannot handle the offsets
    """
    offsets = offsets if offsets is not None else [0] * len(reference_shape)
    # Create an array of zeros with the reference shape
    result = np.zeros(reference_shape)
    # Create a list of slices from offset to offset + shape in each dimension
    insertHere = [slice(offsets[dim], offsets[dim] + array.shape[dim])
                  for dim in range(array.ndim)]
    # Insert the array in the result at the specified offsets
    result[insertHere] = array
    return result
