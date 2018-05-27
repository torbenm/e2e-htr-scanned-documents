import dataset
import util
import os
import cv2
import numpy as np
import tensorflow as tf

FOLDER_NAME = "iam"
BASE_FOLDER = "data"


class IamDataset(dataset.Dataset):

    def __init__(self, binarize, width, height):
        self._binarize = binarize
        self._width = width
        self._height = height
        self._loaded = False
        self._channels = 1
        self._basepath, self._targetpath, self._targetimagepath = get_targetpath(
            binarize, width, height)
        self.preload()

    def maxLength(self):
        return self._maxlength

    def preload(self):
        self._vocab = util.load(self._targetpath, "vocab")
        self._vocab_length = len(self._vocab[0])
        self._lines = util.load(self._targetpath, "lines")
        self._maxlength = max(
            map(lambda x: len(x["text"]), self._lines))

    def compile(self, text):
        length = len(text)
        parsed = [self._vocab[1][c] for c in text]
        parsed.extend([-1] * (self._maxlength - length))
        return parsed

    def decompile(self, values):
        def getKey(key):
            try:
                return self._vocab[0][str(c)]
            except KeyError:
                return ''
        return ''.join([getKey(c) for c in values])

    def _loaddata(self):
        if not self._loaded:
            X = []
            Y = []
            L = []
            for line in self._lines:
                x, y, l = self.loadline(line)
                if x is not None:
                    X.append(x)
                    Y.append(y)
                    L.append(l)
            self._raw_x = 1 - np.asarray(X)
            self._raw_y = np.asarray(Y)
            self._raw_l = np.asarray(L)
            self._loaded = True

    def loadline(self, line):
        l = len(line["text"]) * 2 + 1
        y = np.asarray(self.compile(line["text"]))
        x = cv2.imread(os.path.join(self._targetimagepath, line[
                       "name"] + ".png"), cv2.IMREAD_GRAYSCALE)
        # x = cv2.normalize(x, x, alpha=0, beta=1,
        #                  norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        try:
            x = np.transpose(x, [1, 0])
        except ValueError:
            return None, None, None
        x = np.reshape(x, [self._width, self._height, 1])
        return x, y, l

    def generateBatch(self, batch_size, max_batches=0):
        num_batches = self.getBatchCount(batch_size, max_batches)
        for b in range(num_batches - 1):
            x = self._raw_x[b * batch_size:(b + 1) * batch_size]
            y = self._raw_y[b * batch_size:(b + 1) * batch_size]
            l = self._raw_l[b * batch_size:(b + 1) * batch_size]
            yield x, y, l
        pass

    def prepareDataset(self, validation_batches=0, test_batches=0, batch_size=0, shuffle=False):
        self._loaddata()
        length = len(self._raw_x)
        if shuffle:
            perm = np.random.permutation(length)
            self._raw_x = self._raw_x[perm]
            self._raw_y = self._raw_y[perm]
            self._raw_l = self._raw_l[perm]
        val_length = validation_batches * batch_size
        print "Length of validation set:", val_length
        test_length = test_batches * batch_size
        self._val_x = self._raw_x[0:val_length]
        self._val_y = self._raw_y[0:val_length]
        self._val_l = self._raw_l[0:val_length]
        self._test_x = self._raw_x[val_length:test_length + val_length]
        self._test_y = self._raw_y[val_length:test_length + val_length]
        self._test_l = self._raw_l[val_length:test_length + val_length]

        self._raw_x = self._raw_x[test_length + val_length:]
        self._raw_y = self._raw_y[test_length + val_length:]
        self._raw_l = self._raw_l[test_length + val_length:]
        print "Length of data set:", len(self._raw_x)

    def getValidationSet(self):
        return self._val_x, self._val_y, self._val_l

    def getBatchCount(self, batch_size, max_batches=0):
        total_len = len(self._raw_x)
        num_batches = total_len // batch_size
        return min(
            num_batches, max_batches) if max_batches > 0 else num_batches

    def generateEpochs(self, batch_size, num_epochs, max_batches=0):
        for e in range(num_epochs):
            yield self.generateBatch(batch_size, max_batches=max_batches)
