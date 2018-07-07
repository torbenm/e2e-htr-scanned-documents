import cv2
import numpy as np

img = cv2.imread("./twoliner.png")
vis = img.copy()
size = int((img.shape[0] * img.shape[1])/3*4)
mser = cv2.MSER_create(_delta=5, _max_area=size)


img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
mean = np.mean(img, axis=(0, 1))
img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                            cv2.THRESH_BINARY_INV, 11, mean/4)

img = cv2.erode(img, np.ones((5, 1)))
img = cv2.dilate(img, np.ones((1, 1000)))

img = cv2.copyMakeBorder(
    img, img.shape[0], img.shape[0], img.shape[0], img.shape[0], cv2.BORDER_CONSTANT, value=(0))
vis = cv2.copyMakeBorder(
    vis, vis.shape[0], vis.shape[0], vis.shape[0], vis.shape[0], cv2.BORDER_CONSTANT, value=(0, 0, 0))

cv2.imshow('eroded', img)
cv2.waitKey(0)


regions, _ = mser.detectRegions(img)
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
    cv2.drawContours(vis, [box], 0, (0, 0, 255), 1)


cv2.imshow('text_only', vis)

cv2.waitKey(0)
