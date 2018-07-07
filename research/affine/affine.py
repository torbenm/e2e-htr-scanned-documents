import cv2
import numpy as np

img = cv2.imread("./input.png", cv2.IMREAD_GRAYSCALE)


def translate(tx, ty):
    return np.float32([
        [1, 0, tx],
        [0, 1, ty]
    ])


def rotate(img, alpha):
    rows, cols = img.shape
    M = cv2.getRotationMatrix2D((cols/2, rows/2), alpha, 1)
    print(M)
    return cv2.getRotationMatrix2D((cols/2, rows/2), alpha, 1)


def scale(fx, fy):
    return np.float32([
        [fx, 0, 0],
        [0, fy, 0]
    ])


def shear(sx, sy):
    return np.float32([
        [1, sx, 0],
        [sy, 1, 0]
    ])


img = cv2.warpAffine(img, scale(2, 1), (img.shape[1], img.shape[0]))

cv2.imwrite("./output.png", img)
