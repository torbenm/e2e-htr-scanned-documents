import cv2
import numpy as np
from skimage.filters import threshold_sauvola
from lib.segmentation.Region import Region
from lib.Configuration import Configuration


DEFAULTS = {
    "sauvola_window": 19,
    "erode_kernel": [25, 25],
    "dilate_kernel": [30, 30],
    "scaling": 5,
    "expand": 0
}


class ParagraphDetector():

    def __init__(self, config={}):
        self.config = Configuration(config, DEFAULTS)

    def _enhance(self, img):
        img = cv2.dilate(img, np.ones(
            (self.config["dilate_kernel"][0], self.config["dilate_kernel"][1])))
        img = cv2.erode(img, np.ones(
            (self.config["erode_kernel"][0], self.config["erode_kernel"][1])))
        return img

    def _get_paragraphs(self, img):
        _, contours, _ = cv2.findContours(
            img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return [cv2.boundingRect(contour) for contour in contours], contours

    def extract(self, img):
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        original = img.copy()
        img = cv2.resize(
            img, (int(img.shape[1]*1/self.config["scaling"]), int(img.shape[0]*1/self.config["scaling"])))
        img = 255-self._threshold(img)
        img = self._enhance(img)
        paragraphs, _ = self._get_paragraphs(img)
        return self._extract_regions(original, paragraphs)

    def _extract_region(self, img, region):
        expand = self.config["expand"]*self.config["scaling"]
        x, y, w, h = [a*self.config["scaling"] for a in region]
        return Region(
            pos=(x-expand, y-expand), size=(w+expand*2, h+expand*2), img=img[y:y+h, x:x+w])

    def _extract_regions(self, img, regions):
        return [self._extract_region(img, region) for region in regions]

    def _threshold(self, img):
        thresh_sauvola = threshold_sauvola(
            img, window_size=self.config["sauvola_window"])
        return np.uint8((img > thresh_sauvola)*255)
