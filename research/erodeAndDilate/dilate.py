import cv2
import numpy as np


def _thin(img, width, invert=False):
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (width, width))
    if invert:
        return cv2.erode(img, kernel)
    else:
        return cv2.dilate(img, kernel)


def _thicken(img, width, invert=False):
    return _thin(img, width, not invert)


image = cv2.imread('input3.png', cv2.IMREAD_GRAYSCALE)

image = _thin(image, 3)

cv2.imwrite("output.png", image)
