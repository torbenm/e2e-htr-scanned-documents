import cv2
import numpy as np


def binarize(images):
    return [_binarize(image) for image in images]


def _binarize(image):
    threshold = np.mean(image, axis=(0, 1, 2))
    return cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
