import cv2
import numpy as np
import AmbrosioTortorelliMinimizer

img = cv2.imread("./images/rl01.jpg")

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
mean = np.mean(img, axis=(0, 1))
img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                            cv2.THRESH_BINARY, 51, mean/4)

img = cv2.GaussianBlur(img, (51, 21), 0)
cv2.imshow('blurred', img)
cv2.waitKey(-1)

result, edges = [], []
for channel in cv2.split(img):
    solver = AmbrosioTortorelliMinimizer.AmbrosioTortorelliMinimizer(
        channel, iterations=1, tol=0.1, solver_maxiterations=6)
    f, v = solver.minimize()
    result.append(f)
    edges.append(v)

f = cv2.merge(result)
v = edges[0]
cv2.imshow("edges", v)
cv2.waitKey(-1)
cv2.imshow("image", f)
cv2.waitKey(-1)
