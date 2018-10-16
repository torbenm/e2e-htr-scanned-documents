from . import util
from .Dataset import Dataset
from lib.Configuration import Configuration
import os
import math
import numpy as np
import cv2
import sys
from random import shuffle
from data.steps.pipes import warp, morph, convert
from .Dataset import Dataset
from data.steps.pipes import crop, threshold, invert, padding
from data.ImageAugmenter import ImageAugmenter


class RegionDataset(Dataset):

    def __init__(self, regions, model_path, data_config={}):
        self.model_path = model_path
        self._load_vocab()
        self._load_meta()
        self._scaling = 1.0
        self._max_height = 10000
        self._max_width = 10000
        self.set_regions(regions)
        self.data_config = Configuration({
            "preprocess": {
                "crop": True,
                "invert": True,
                "scale": 1,
                "padding": 5
            }
        })
        self.augmenter = ImageAugmenter(self.data_config)

    def info(self):
        self.meta('Dataset Configuration')

    def scaling(self, scaling, max_height, max_width):
        self.augmenter.config['preprocess.scale'] = scaling
        self._max_height = max_height
        self._max_width = max_width

    def _load_meta(self):
        self.meta = Configuration(util.loadJson(self.model_path, "data_meta"))

    def _load_vocab(self):
        self.vocab = util.loadJson(self.model_path, "vocab")
        self.vocab_length = len(self.vocab[0])

    def _load_sets(self):
        self.data = np.asarray(list(filter(lambda x: x is not None, [
            self._loadimage(region) for region in self.regions])))

    def _loadimage(self, region):
        if len(region.img.shape) > 2:
            img = cv2.cvtColor(region.img, cv2.COLOR_BGR2GRAY)
        else:
            img = region.img
        target_size = (
            int(self.meta["height"] -
                (self.data_config.default('preprocess.padding', 0)*2)),
            int(self.meta["width"] -
                (self.data_config.default('preprocess.padding', 0)*2))
        )
        img = self.augmenter.preprocess(img, target_size)
        img = self.augmenter.postprocesss(img)
        if img is None:
            img = np.zeros((self.meta["height"], self.meta["width"]))
        return self.augmenter.add_graychannel(img)

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
        batch_data = np.asarray(self.data[index *
                                          batch_size:min((index+1)*batch_size, len(self.data))])
        if with_filepath:
            return batch_data, [], [], []
        else:
            return batch_data, [], []

    def generateBatch(self, batch_size=0, max_batches=0, dataset="", with_filepath=False):
        num_batches = self.getBatchCount(batch_size, max_batches, "")
        for b in range(num_batches):
            yield self._load_batch(b, batch_size, "", with_filepath)
        pass

    def generateEpochs(self, batch_size, num_epochs, max_batches=0, dataset="train", with_filepath=False):
        return [self.generateBatch()]

    def getBatchCount(self, batch_size, max_batches=0, dataset=""):
        return int(np.ceil(len(self.data)/float(batch_size)))
