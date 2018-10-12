import numpy as np
import cv2
import math
import peakutils
from skimage.filters import threshold_sauvola
from scipy.signal import argrelextrema


from lib.Configuration import Configuration
from lib.segmentation.AStarPathFinder import AStarPathFinder
from lib.segmentation.Region import Region
from data.steps.pipes.crop import _crop

DEFAULT_CONFIG = {
    "sauvola_window": 19,
    "relative_maxima_dist": 56,
    "scale_factor": 4.0,
    "astar": {}
}


class AStarLineSegmentation(object):

    def __init__(self, config={}):
        self.config = Configuration(config, DEFAULT_CONFIG)

    def _binarize(self, img):
        thresh_sauvola = threshold_sauvola(
            img, window_size=self.config["sauvola_window"])
        return np.uint8((img > thresh_sauvola)*255)

    def _ink_density(self, img, axis=1):
        return np.mean(img, axis=1, dtype=np.int32)

    def _local_maxima(self, y_hist):
        min_dist = self.config["relative_maxima_dist"] / \
            float(self.config["scale_factor"])
        min_val = np.mean(y_hist) - np.std(y_hist)
        maxima = peakutils.indexes(y_hist, thres=min_val,
                                   min_dist=min_dist, thres_abs=True)
        return maxima

    def _local_minima(self, y_hist):
        max_val = np.max(y_hist) - np.std(y_hist)
        print(max_val)
        minima = argrelextrema(y_hist, np.less_equal)
        return np.int32(list(filter(lambda x: x < max_val, *minima)))

    def _line_start_from_maxima(self, maxima):
        return np.int32((maxima[1:] + maxima[:-1]) / 2)

    def _enhance(self, img):
        return cv2.GaussianBlur(img, (3, 3), 0)

    def _exec_astar(self, img, starting_points):
        astar = AStarPathFinder(img, self.config["astar"])
        return [self._resize_array(astar.find_path(start)) for start in starting_points]

    def _resize_array(self, a):
        return [(int(x*self.config["scale_factor"]), int(y*self.config["scale_factor"])) for x, y in a]

    def _extract_line(self, img, line):
        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        cv2.fillPoly(mask, np.array([line]), 1)

        mask_applied = cv2.bitwise_and(255-img, 255-img, mask=mask)
        cropped = _crop(mask_applied)
        if cropped is None:
            return None
        return Region(path=line, img=255 - cropped)

    def _extract_lines(self, img, line_paths):
        lines = []
        previous = [(0, 0), (img.shape[1], 0)]
        for path in line_paths:
            lines.append(self._extract_line(img, [*reversed(previous), *path]))
            previous = path
        lines.append(self._extract_line(
            img, [*reversed(previous), (0, img.shape[0]), (img.shape[1], img.shape[0])]))
        return list(filter(lambda x: x is not None, lines))

    def __call__(self, img):
        original = img
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        scaling = 1.0/self.config["scale_factor"]
        img = cv2.resize(
            img, (int(img.shape[1] * scaling), int(img.shape[0] * scaling)))
        img = 255 - self._binarize(img)
        # TODO: make this conditional
        img = self._enhance(img)
        y_hist = self._ink_density(img)
        maxima = self._local_maxima(y_hist)
        starting_points = self._line_start_from_maxima(maxima)
        # starting_points = self._local_minima(y_hist)
        # print(starting_points)
        lines = self._exec_astar(img, starting_points)
        return self._extract_lines(original, lines), lines
