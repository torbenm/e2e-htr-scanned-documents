import cv2
import numpy as np


def com_stats(A, axis=0):
    print(A)
    A = A.astype(float)    # if you are worried about int vs. float
    n = A.shape[axis]
    m = A.shape[(axis-1) % 2]
    r = np.arange(1, n+1)
    R = np.vstack([r] * m)
    if axis == 0:
        R = R.T

    mu = np.average(R, axis=axis, weights=A)
    var = np.average(R**2, axis=axis, weights=A) - mu**2
    std = np.sqrt(var)
    return mu, var, std


image = cv2.imread('input3.png', cv2.IMREAD_GRAYSCALE)

mu, var, std = com_stats(image)
print(len(mu))
mu = int(np.average(mu))
std = int(np.median(std))
var = int(np.median(var))

diff = std


image[mu, :] = 0
image[mu+diff, :] = 0
image[mu-diff, :] = 0

print(np.average(mu), var, std)

cv2.imwrite("output.png", image)
