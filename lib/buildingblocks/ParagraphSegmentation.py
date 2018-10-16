import numpy as np

from lib.Configuration import Configuration
from lib.segmentation.ParagraphRegionExtractor import ParagraphRegionExtractor

DEFAULTS = {
    "extractor": {}
}
# TODO: configuration via config


class ParagraphSegmentation(object):

    def __init__(self, config={}):
        self.config = Configuration(config, DEFAULTS)
        self.region_extractor = ParagraphRegionExtractor(
            self.config["extractor"])

    def __call__(self, img):
        return self.region_extractor.extract(img)

    def close(self):
        pass
