import cv2


def load(src, config, context={}):
    if config == "greyscale" or config == "grayscale":
        return [cv2.imread(src, cv2.IMREAD_GRAYSCALE)]
    else:
        return [cv2.imread(src)]
