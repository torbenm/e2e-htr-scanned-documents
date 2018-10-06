import numpy as np

from lib.Configuration import Configuration
from lib.segmentation.WordRegionExtractor import WordRegionExtractor


DEFAULTS = {
    "extractor": {}
}


class WordSegmentation(object):

    def __init__(self, config={}):
        self.config = Configuration(config, DEFAULTS)
        self.region_extractor = WordRegionExtractor(
            self.config["extractor"])

    def __call__(self, img):
        return self.region_extractor.extract(img)

    def close(self):
        pass


if __name__ == "__main__":
    import cv2

    img = cv2.imread(
        "../paper-notes/data/final/train/00009-paper.png")
    lineseg = WordSegmentation()
    lines = lineseg(img)
    for line in lines:
        cv2.rectangle(
            img, line.pos, (line.pos[0]+line.size[0], line.pos[1]+line.size[1]),  (0, 195, 0), 1)
        # cv2.imshow('line', line.img)
        # cv2.waitKey(0)
    cv2.imwrite("prediction/wordsegex/ex_9.png", img)
