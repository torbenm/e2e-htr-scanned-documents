import numpy as np

from lib.Configuration import Configuration
from lib.segmentation.ParagraphDetector import ParagraphDetector

DEFAULTS = {
    "extractor": {}
}


class ParagraphDetection(object):

    def __init__(self, config={}):
        self.config = Configuration(config, DEFAULTS)
        self.region_extractor = ParagraphDetector(
            self.config["extractor"])

    def __call__(self, img, file):
        return self.region_extractor.extract(img)

    def close(self):
        pass
