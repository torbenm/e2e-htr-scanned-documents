import os
import cv2
import numpy as np


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


def getlineheight(image):
    _min = boundary(image)
    _max = boundary(image, flip=True)
    return max(_max - _min, 0)


path = "/Users/torbenmeyer/Development/masterthesis/tensorflow-htr/data/output/real-iam-lines-hfac/imgs"
heights = []
for filename in os.listdir(path):
    if filename.endswith(".png"):
        image = cv2.imread(os.path.join(path, filename), cv2.IMREAD_GRAYSCALE)
        height = getlineheight(image)
        heights.append(height)
        print("{:50}{}".format(filename, height))

print(np.max(heights))
print(np.min(heights))
print(np.average(heights))
print(np.median(heights))
