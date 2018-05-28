import util
import os
import math
import numpy as np
import cv2


class Dataset(object):

    def __init__(self, name):
        self.name = name
        self.datapath = os.path.join(util.OUTPUT_PATH, name)
        self._load_vocab()
        self._load_meta()
        self._load_sets()
        self._calc_max_length()
        self._compile_sets()
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
                return self.vocab[0][str(c)]
            except KeyError:
                return ''
        return ''.join([getKey(c) for c in values])

    def _loadline(self, line):
        l = len(line["compiled"])
        y = np.asarray(line["compiled"])
        x = cv2.imread(line["path"], cv2.IMREAD_GRAYSCALE)
        try:
            x = np.transpose(x, [1, 0])
        except ValueError:
            return None, None, None
        x = np.reshape(x, [self.meta["width"], self.meta["height"], 1])
        return x, y, l

    def _load_batch(self, index, batch_size, dataset):
        X = []
        Y = []
        L = []
        for idx in range(index * batch_size, min((index + 1) * batch_size, len(self.data[dataset]))):
            x, y, l = self._loadline(self.data[dataset][idx])
            if x is not None:
                X.append(x)
                Y.append(y)
                L.append(l)
        return np.asarray(X), np.asarray(Y), np.asarray(L)

    def generateBatch(self, batch_size, max_batches=0, dataset="train"):
        num_batches = self.getBatchCount(batch_size, max_batches, dataset)
        for b in range(num_batches):
            yield self._load_batch(b, batch_size, dataset)
        pass

    def generateEpochs(self, batch_size, num_epochs, max_batches=0, dataset="train"):
        for e in range(num_epochs):
            yield self.generateBatch(batch_size, max_batches=max_batches, dataset=dataset)

    def getBatchCount(self, batch_size, max_batches=0, dataset="train"):
        total_len = len(self.data[dataset])
        num_batches = int(math.ceil(float(total_len) / batch_size))
        return min(
            num_batches, max_batches) if max_batches > 0 else num_batches
