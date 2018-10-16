import numpy as np

from lib.Configuration import Configuration
from lib.segmentation.AStarLineSegmentation import AStarLineSegmentation
from lib.segmentation.ParagraphRegionExtractor import ParagraphRegionExtractor


DEFAULTS = {
    "extractor": {}
}

# TODO: AStarPathFinder config


class LineSegmentation(object):

    def __init__(self, config={}):
        self.config = Configuration(config, DEFAULTS)
        self.astar = AStarLineSegmentation(self.config["extractor"])

    def __call__(self, regions):
        lines = []
        for region in regions:
            region_lines, _ = self.astar(region.img)
            [region_line.translate(
                region.pos) for region_line in region_lines]
            lines.extend(region_lines)
        return lines

    def close(self):
        pass


if __name__ == "__main__":
    import cv2

    img = cv2.imread(
        "../paper-notes/data/final/train/00009-stripped.png")
    lineseg = LineSegmentation()
    lines = lineseg(img)
    for line in lines:
        cv2.drawContours(img, [np.array(line.path)], 0, (0, 195, 0), 1)
        # cv2.imshow('line', line.img)
        # cv2.waitKey(0)
    cv2.imwrite("prediction/linesegex/ex_9.png", img)
