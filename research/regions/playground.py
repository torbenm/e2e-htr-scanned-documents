
# Based on
# https://github.com/HsiehYiChia/Scene-text-recognition
#  and
"""
Cho, Hojin, Myungchul Sung, and Bongjin Jun.
"Canny text detector: Fast and robust scene text localization algorithm."
Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016.
"""
# %%##########################################
#   Initialize Vars                         #
#############################################
import os
print("Working in", os.getcwd())

import cv2
import numpy as np
from matplotlib import pyplot as plt


def image_of_contour(img, contour):
    x, y, w, h = cv2.boundingRect(contour)
    mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)
    crop = img[y:y+h, x:x+w, :]  # cv2.bitwise_and(img[y:y+h, x:x+w, :],
    # img[y:y+h, x:x+w, :], mask=mask[y:y+h, x:x+w, :])
    cv2.imshow('croppart', crop)
    cv2.waitKey(0)


IMAGE_PATH = "./scan.jpg"

img = cv2.imread(IMAGE_PATH)
img = cv2.resize(img, (533, 800))

mser = cv2.MSER_create(_delta=5)


vis = img.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

mean = np.mean(gray, axis=(0, 1))
gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                             cv2.THRESH_BINARY_INV, 11, mean/4)

gray = cv2.dilate(gray, np.ones((1, 10)))

cv2.imshow('pre-mser', gray)
cv2.waitKey(0)

regions, _ = mser.detectRegions(gray)
hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
# cv2.polylines(vis, hulls, 1, (0, 255, 0))
# cv2.imshow('img', vis)
# cv2.waitKey(0)

mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)

for contour in hulls:
    # x, y, w, h = cv2.boundingRect(contour)
    # if h > w:
    #     continue
    # image_of_contour(vis, contour)
    # cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(vis, [box], 0, (255, 0, 0), 1)


# this is used to find only text regions, remaining are ignored
# text_only = cv2.bitwise_and(img, img, mask=mask)

cv2.imshow('text_only', vis)

cv2.waitKey(0)
