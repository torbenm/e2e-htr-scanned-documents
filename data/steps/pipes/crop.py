import numpy as np


def crop(images):
    return [_crop(image) for image in images]


def _crop(image):
    bbox = bounding_box(image)
    if bbox is None:
        return None
    return image[bbox[0]:bbox[2], bbox[1]:bbox[3]]


def first_nonzero(arr, axis, invalid_val=-1):
    mask = arr != 0
    im2 = np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)
    return np.min(im2[im2 != invalid_val])


def last_nonzero(arr, axis, invalid_val=-1):
    mask = arr != 0
    val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    im2 = np.where(mask.any(axis=axis), val, invalid_val)
    return np.max(im2[im2 != invalid_val])


def bounding_box(image):
    try:
        top = first_nonzero(image, 0)
        left = first_nonzero(image, 1)
        bottom = last_nonzero(image, 0)
        right = last_nonzero(image, 1)
        if bottom <= top or right <= left:
            return None
        return top, left, bottom, right
    except ValueError:
        return None
