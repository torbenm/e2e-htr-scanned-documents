import cv2

img = cv2.imread("../thesis/Figs/datasets/borderregions.png",
                 cv2.IMREAD_GRAYSCALE)
img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
cv2.imwrite("../thesis/Figs/datasets/borderregions_color.png", img)
