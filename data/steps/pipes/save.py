import os
import cv2


def save(images, filename, target):
    idx = 0
    paths = []
    for image in images:
        paths.append(_save(image, idx, filename, target))
        idx = idx + 1
    return paths


def _save(image, index, filename, target):
    ipath = os.path.join(target, "{}_{}{}".format(
        filename[0], index, filename[1]))
    cv2.imwrite(ipath, image)
    return ipath
