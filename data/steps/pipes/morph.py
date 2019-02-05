import numpy as np
import cv2


def morph(images, config, invert=False):
    altered_images = []
    for image in images:
        altered_images.append(_random_morph(image, config, invert))
    return altered_images


def _random_morph(image, config, invert=False, return_op=False):
    if np.random.uniform() < config['prob']:
        op = int(np.random.uniform(0, len(config['ops'])))
        morphed = _morph(image, config['ops'][op]['name'],
                         config['ops'][op]['values'], invert)
        if return_op:
            return morphed, config['ops'][op]['name'], config['ops'][op]['values']
        else:
            return morphed
    else:
        if return_op:
            return image, None, None
        else:
            return image


def _morph(image, op_name, op_values, invert=False):
    width = int(np.random.uniform(op_values[0], op_values[1]+1))
    height = int(np.random.uniform(op_values[0], op_values[1]+1))
    return MORPH_OPS[op_name](image, width, height, invert)


def _thin(img, width, height, invert=False):
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (height, width))
    if invert:
        return cv2.erode(img, kernel)
    else:
        return cv2.dilate(img, kernel)


def _thicken(img, width, height, invert=False):
    return _thin(img, width, height, not invert)


MORPH_OPS = {
    "thin": _thin,
    "thicken": _thicken
}
