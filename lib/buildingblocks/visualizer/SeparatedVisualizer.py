import numpy as np
import cv2

from lib.Configuration import Configuration

DEFAULT_CONFIG = {}


class SeparatedVisualizer(object):

    def __init__(self, config={}):
        self.config = Configuration(config, DEFAULT_CONFIG)

    def __call__(self, original, merged):
        if len(original.shape) > 2 and original.shape[2] == 3:
            original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        return np.concatenate((original, merged), axis=1)
