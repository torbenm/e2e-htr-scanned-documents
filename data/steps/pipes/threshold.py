import cv2
import numpy as np


def threshold(images, invert=False):
    return [_threshold(image, invert) for image in images]


def _threshold(image, invert):
    threshold = np.mean(image, axis=(0, 1))
    _, res_img = cv2.threshold(
        image, threshold, 255, (cv2.THRESH_TOZERO if invert else cv2.THRESH_TOZERO_INV) + cv2.THRESH_OTSU)
    return res_img
