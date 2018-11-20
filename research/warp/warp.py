import cv2
from data.steps.pipes.warp import _warp


img = cv2.imread("./research/warp/example.png", cv2.IMREAD_GRAYSCALE)
img = _warp(img, [30, 30], 2.7)
cv2.imwrite("./research/warp/output.png", img)
