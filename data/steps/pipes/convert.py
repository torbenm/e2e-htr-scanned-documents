import numpy as np


def pil2cv2(images):
    return [_pil2cv2(image) for image in images]


def _pil2cv2(image):
    npa = np.array(image.convert('RGB'))
    return npa[:, :, ::-1].copy()
