import cv2
import numpy as np


def threshold(images, invert=False):
    return [_threshold(image, invert) for image in images]


def _threshold(image, invert=False):
    if not invert:
        image = 255 - image
    threshold = np.mean(image, axis=(0, 1))
    _, res_img = cv2.threshold(
        image, threshold, 255, cv2.THRESH_TOZERO + cv2.THRESH_OTSU)
    if not invert:
        res_img = 255 - res_img
    return res_img
