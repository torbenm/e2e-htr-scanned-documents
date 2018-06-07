import cv2
import numpy as np


def first_nonzero(arr, axis, invalid_val=-1):
    mask = arr != 0
    im2 = np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)
    return np.min(im2[im2 != invalid_val])


def last_nonzero(arr, axis, invalid_val=-1):
    mask = arr != 0
    val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    im2 = np.where(mask.any(axis=axis), val, invalid_val)
    return np.max(im2[im2 != invalid_val])


def bounding_box(image):
    return first_nonzero(image, 0), first_nonzero(image, 1), last_nonzero(image, 0), last_nonzero(image, 1)

image = cv2.imread('input.png')
image = 255 - image
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
threshold = np.mean(image, axis=(0, 1))
_, image = cv2.threshold(
    image, threshold, 255, cv2.THRESH_TOZERO + cv2.THRESH_OTSU)

bbox = bounding_box(image)
image = image[bbox[0]:bbox[2], bbox[1]:bbox[3]]

cv2.imwrite("output.png", image)
