import numpy as np
from PIL import Image
import cv2


def pil2cv2(images, mode='L'):
    return [_pil2cv2(image, mode) for image in images]


def _pil2cv2(image, mode='L'):
    npa = np.array(image.convert(mode))
    return npa[:, :, ::-1].copy() if mode == 'RGB' else npa


def cv2pil(images, mode='L'):
    return [_cv2pil(image, mode) for image in images]


def _cv2pil(image, mode='L'):
    if mode == 'RGB':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image, mode=mode)
