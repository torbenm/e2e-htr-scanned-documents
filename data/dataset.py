from . import util
from config.config import Configuration
import os
import math
import numpy as np
import cv2
import sys


class Dataset(object):

    def __init__(self, name, transpose=True, dynamic_width=False):
        self.name = name
        self.dynamic_width = dynamic_width
        self.min_width_factor = 15
        self.datapath = os.path.join(util.OUTPUT_PATH, name)
        self._load_vocab()
        self._load_meta()
        self._load_sets()
        self._calc_max_length()
        self._compile_sets()
        self.transpose = transpose
        self.channels = 1
        self._fill_meta()

    def info(self):
        self.meta('Dataset Configuration')

    def _load_meta(self):
        self.meta = Configuration(util.loadJson(self.datapath, "meta"))

    def _load_vocab(self):
        self.vocab = util.loadJson(self.datapath, "vocab")
        self.vocab_length = len(self.vocab[0])

    def _fill_meta(self):
        self.meta['vocab.size'] = self.vocab_length
        self.meta['train.count'] = len(self.data['train'])
        self.meta['train.count'] = len(self.data['train'])
        self.meta['dev.count'] = len(self.data['dev'])
        self.meta['test.count'] = len(self.data['test'])
        if 'print_train' in self.data:
            self.meta['print_train.count'] = len(self.data['print_train'])
            self.meta['print_dev.count'] = len(self.data['print_dev'])
            self.meta['print_test.count'] = len(self.data['print_test'])

    def _load_sets(self):
        self.data = {
            "train": util.loadJson(self.datapath, "train"),
            "dev": util.loadJson(self.datapath, "dev"),
            "test": util.loadJson(self.datapath, "test")
        }
        if self.meta.default('printed', False):
            self.data['print_train'] = util.loadJson(
                self.datapath, "print_train")
            self.data['print_dev'] = util.loadJson(self.datapath, "print_dev")
            self.data['print_test'] = util.loadJson(
                self.datapath, "print_test")
        if self.dynamic_width:
            self._sort_by_width("train")
            self._sort_by_width("dev")
            if self.meta.default('printed', False):
                self._sort_by_width("print_train")
                self._sort_by_width("print_dev")

    def _sort_by_width(self, dataset):
        print("Sorting {} dataset by width...".format(dataset))
        for datapoint in self.data[dataset]:
            img = cv2.imread(datapoint["path"], cv2.IMREAD_GRAYSCALE)
            datapoint["width"] = img.shape[1]
        self.data[dataset].sort(key=lambda x: x["width"])

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

    def load_image(self, path, transpose=False):
        x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if transpose:
            try:
                x = np.transpose(x, [1, 0])
                if self.dynamic_width:
                    return np.reshape(x, [x.shape[0], x.shape[1], 1])
            except ValueError:
                return None, None, None, None
            if x.shape[0] != self.meta["width"] or x.shape[1] != self.meta["height"]:
                x = pad(x, (self.meta["width"], self.meta["height"]))
            x = np.reshape(x, [self.meta["width"], self.meta["height"], 1])
        else:
            if self.dynamic_width:
                return np.reshape(x, [x.shape[0], x.shape[1], 1])
            if x.shape[1] != self.meta["width"] or x.shape[0] != self.meta["height"]:
                x = pad(x, (self.meta["height"], self.meta["width"]))
            x = np.reshape(x, [self.meta["height"], self.meta["width"], 1])
        return x

    def _loadline(self, line, transpose=True):
        l = len(line["truth"])
        y = np.asarray(line["compiled"])
        x = self.load_image(line["path"])
        return x, y, l, line["path"]

    def _loadprintline(self, line, transpose=True):
        y = line["truth"]
        x = self.load_image(line["path"])
        return x, [y], 0, line["path"]

    def _load_batch(self, index, batch_size, dataset, with_filepath=False):
        X = []
        Y = []
        L = []
        F = []

        parseline = self._loadline if not dataset.startswith(
            "print_") else self._loadprintline

        for idx in range(index * batch_size, min((index + 1) * batch_size, len(self.data[dataset]))):
            x, y, l, f = parseline(
                self.data[dataset][idx], self.transpose)
            if x is not None:
                X.append(x)
                Y.append(y)
                L.append(l)
                F.append(f)

        if self.dynamic_width:
            batch_width = max(
                np.max(list(map(lambda _x: _x.shape[1], X))), max(L)*self.min_width_factor)
            print(batch_width, max(L))
            X_ = np.zeros(
                (len(X), self.meta["height"], batch_width, 1), dtype=np.int32)
            for idx, x in enumerate(X):
                X_[idx, 0:x.shape[0], 0:x.shape[1], :] = x
            X = X_
        else:
            X = np.asarray(X)
        if not with_filepath:
            return X, np.asarray(Y), np.asarray(L)
        else:
            return X, np.asarray(Y), np.asarray(L), F

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
