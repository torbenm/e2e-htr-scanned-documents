import numpy as np
import cv2
import os

from lib.Configuration import Configuration

DEFAULT_CONFIG = {}


class SeparatedVisualizer(object):

    def __init__(self, config={}):
        self.config = Configuration(config, DEFAULT_CONFIG)

    def __call__(self, original, merged):
        if len(original.shape) > 2 and original.shape[2] == 3:
            original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        return np.concatenate((original, merged), axis=1)

    def store(self, vizimage, original_file):
        if self.config.default("store", False):
            os.makedirs(self.config["store"], exist_ok=True)
            filename = os.path.basename(original_file)
            cv2.imwrite(os.path.join(self.config["store"], filename), vizimage)

    def store(self, vizimage, original_file):
        if self.config.default("store", False):
            os.makedirs(self.config["store"], exist_ok=True)
            filename = os.path.basename(original_file)
            cv2.imwrite(os.path.join(self.config["store"], filename), vizimage)
