import cv2
from data.steps.pipes.warp import _warp
from data.steps.pipes.convert import _cv2pil, _pil2cv2


img = cv2.imread("./research/warp/example.png", cv2.IMREAD_GRAYSCALE)

for i in range(0, img.shape[0], 30):
    img[i, :] = 127
for i in range(0, img.shape[1], 30):
    img[:, i] = 127
cv2.imwrite("./research/warp/warp_before.png", img)
img = _cv2pil(255-img)
img = _warp(img, [30, 30], 2.7)
img = 255-_pil2cv2(img)
cv2.imwrite("./research/warp/warp_after.png", img)
