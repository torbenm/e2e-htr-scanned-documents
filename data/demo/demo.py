import cv2
import numpy as np


image = cv2.imread('input2.png', cv2.IMREAD_GRAYSCALE)

# threshold


def _threshold(image, invert=False):
    threshold = np.mean(image, axis=(0, 1))
    _, res_img = cv2.threshold(
        image, threshold, 255,  cv2.THRESH_OTSU)
    return res_img


image = _threshold(image, True)


def boundary(image, axis=0, invalid_val=-1, sparse_val=0, flip=False):
    mask = image != sparse_val
    val = None
    if flip:
        val = image.shape[axis] - \
            np.flip(mask, axis=axis).argmax(axis=axis) - 1
    else:
        val = mask.argmax(axis=axis)
    masked = np.where(mask.any(axis=axis), val, invalid_val)
    result = np.average(masked[masked != invalid_val])
    return result.astype(int)


_min = boundary(image, sparse_val=255)
_max = boundary(image, flip=True, sparse_val=255)

image[_min, :] = 0
image[_max, :] = 0

cv2.imwrite("output.png", image)
