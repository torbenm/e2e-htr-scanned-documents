import cv2
import numpy as np
from skimage.filters import threshold_sauvola


def viz(img):
    cv2.imshow('img', img)
    cv2.waitKey(0)


DEFAULTS = {
    "sauvola_window": 19
}


class Region(object):
    def __init__(self, pos, size, img):
        self.img = img
        self.pos = pos
        self.size = size


class RegionExtractor():

    def __init__(self, config={}):
        self.config = DEFAULTS

    def _enhance(self, img):
        img = cv2.dilate(img, np.ones((30, 30)))
        img = cv2.erode(img, np.ones((25, 25)))
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
            img, (int(img.shape[1]*1/5.0), int(img.shape[0]*1/5.0)))
        img = 255-self._threshold(img)
        img = self._enhance(img)
        paragraphs, _ = self._get_paragraphs(img)
        return self._extract_regions(original, paragraphs)

    def _extract_region(self, img, region):
        x, y, w, h = [a*5 for a in region]
        return Region(
            (x, y), (w, h), img[y:y+h, x:x+w])

    def _extract_regions(self, img, regions):
        return [self._extract_region(img, region) for region in regions]

    def _threshold(self, img):
        thresh_sauvola = threshold_sauvola(
            img, window_size=self.config["sauvola_window"])
        return np.uint8((img > thresh_sauvola)*255)


if __name__ == "__main__":
    IMAGE = "./images/gt02.png"
    img = cv2.imread(IMAGE)
    re = RegionExtractor()
    regions = re.extract(img)

    for region in regions:
        cv2.imshow('final_region', region.img)
        cv2.waitKey(0)
        x, y = region.pos
        w, h = region.size
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
    # cv2.imwrite('output.png', img)
    cv2.imshow('All', cv2.resize(img, (533, 800)))
    cv2.waitKey(0)
