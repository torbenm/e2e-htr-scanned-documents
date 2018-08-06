from . import util
from .Dataset import Dataset
from config.config import Configuration
import os
import math
import numpy as np
import cv2
import sys
from random import shuffle
from data.steps.pipes import warp, morph, convert
from .Dataset import Dataset
from data.steps.pipes import crop, threshold, invert, padding


class RegionDataset(Dataset):

    def __init__(self, regions, model_path, data_config={}):
        self.model_path = model_path
        self._load_vocab()
        self._load_meta()
        self._scaling = 1.0
        self._max_height = 10000
        self._max_width = 10000
        self.set_regions(regions)

    def info(self):
        self.meta('Dataset Configuration')

    def scaling(self, scaling, max_height, max_width):
        self._scaling = scaling
        self._max_height = max_height
        self._max_width = max_width

    def _load_meta(self):
        self.meta = Configuration(util.loadJson(self.model_path, "data_meta"))

    def _load_vocab(self):
        self.vocab = util.loadJson(self.model_path, "vocab")
        self.vocab_length = len(self.vocab[0])

    def _load_sets(self):
        self.data = [self._preprocess(region) for region in self.regions]

    def _scale(self, img, factor):
        height = int(img.shape[0] / factor)
        width = int(img.shape[1] / factor)
        return cv2.resize(img, (width, height))

    def _scale_img(self, img):
        if img.shape[0] == 0 or img.shape[1] == 0:
            print(im.shape)
            return img
        factor = max(img.shape[0] / self._max_height,
                     img.shape[1] / self._max_width,
                     self._scaling)
        img = self._scale(img, factor)
        return img

    def _preprocess(self, region):
        img = cv2.cvtColor(region.img, cv2.COLOR_BGR2GRAY)
        img = invert._invert(img)
        img = threshold._threshold(img, True)
        img = crop._crop(img)
        img = self._scale_img(img)
        img = padding._pad_cv2(img, 5, 0)

        img = pad(img, (self.meta["height"], self.meta["width"]))
        img = np.reshape(img, [self.meta["height"], self.meta["width"], 1])
        return img

    def set_regions(self, regions):
        self.regions = regions
        if regions is not None:
            self._load_sets()

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

    def _load_batch(self, index, batch_size, dataset, with_filepath=False):
        if with_filepath:
            return np.asarray(self.data), [], [], []
        else:
            return np.asarray(self.data), [], []

    def generateBatch(self, batch_size=0, max_batches=0, dataset="", with_filepath=False):
        return [self._load_batch(0, 0, "", with_filepath)]

    def generateEpochs(self, batch_size, num_epochs, max_batches=0, dataset="train", with_filepath=False):
        return [self.generateBatch()]

    def getBatchCount(self, batch_size, max_batches=0, dataset=""):
        return 1  # its always! one


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
