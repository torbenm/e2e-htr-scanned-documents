from . import util
from .Dataset import Dataset
from lib.Configuration import Configuration
import os
import math
import numpy as np
import cv2
import sys
from random import shuffle
from .Dataset import Dataset
from data.Slicer import Slicer
from data.steps.pipes import crop, threshold, invert, padding, warp, morph, convert


class PaperNoteSlicesSingle(Dataset):

    slices = []
    img = None
    vocab = {}
    meta = Configuration({})

    def __init__(self, **kwargs):
        self.slice_width = kwargs.get('slice_width', 320)
        self.slice_height = kwargs.get('slice_height', 320)
        self.slicer = Slicer(**kwargs)

    def load_file(self, filepath):
        self.img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        self.slices = self.slicer(self.img)
        return self.img

    def info(self):
        pass

    def compile(self, text):
        return text

    def decompile(self, values):
        return values

    def merge_slices(self, slices, original_shape):
        return self.slicer.merge(slices, original_shape)

    def generateBatch(self, batch_size=0, max_batches=0, dataset="", with_filepath=False):
        for idx in range(self.getBatchCount(batch_size, max_batches)):
            slices = np.asarray(
                self.slices[(idx*batch_size):((idx+1)*batch_size)])/255.0
            if with_filepath:
                yield slices, [], [], []
            else:
                yield slices, [], []
        pass

    def generateEpochs(self, batch_size, num_epochs, max_batches=0, dataset="train", with_filepath=False):
        return [self.generateBatch()]

    def getBatchCount(self, batch_size, max_batches=0, dataset="train"):
        batch_count = np.ceil(len(self.slices)/batch_size)
        return int(batch_count)
