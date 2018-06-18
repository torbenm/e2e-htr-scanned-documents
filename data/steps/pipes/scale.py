import cv2
import numpy as np
from numpy.lib.stride_tricks import as_strided

"""
Scaling by factor, factor will be extracted:
{
    "mode": "factor",
    "size": [w, h]
}

Scaling by height, width will be variable:
{
    "mode": "height",
    "size": [-1, h]
}
"""


def _get_scale_fn(mode):
    if mode == 'factor':
        return _scale_by_factor
    elif mode == 'height':
        return _scale_by_height
    elif mode == 'line':
        return _scale_by_line_centroid
    return None


def scale(images, config, fill=(255, 255, 255)):
    # scale = 1.0, size = (100, 100), fill = (255, 255, 255)):
    mode = config['mode'] if 'mode' in config else 'factor'
    scale_fn = _get_scale_fn(mode)
    if scale_fn is None:
        return images
    return [scale_fn(image, config, fill) for image in images]


def _scale_by_factor(image, config, fill=(255, 255, 255)):
    width, height = config['size']
    factor = config['factor']
    image, ws, hs = _do_scale(image, factor, max_size=(width, height))
    h_pad = height - hs if height > -1 else 0
    w_pad = width - ws if width > -1 else 0

    return cv2.copyMakeBorder(
        image, 0, h_pad, 0, w_pad, cv2.BORDER_CONSTANT, value=fill)


def _scale_by_height(image, config, fill=(255, 255, 255)):
    target_size = config['size']
    current_height = image.shape[0]
    top
    factor = current_height / float(target_size[1])
    image, _, _ = _do_scale(image, factor, target_size=target_size)
    return image


def _scale_by_line_centroid(image, config, fill=(255, 255, 255)):
    target_size = config['size']
    image_c = image+1
    current_height, top, bottom = _line_height(image_c)
    top = max(top, 0)
    bottom = min(bottom, image.shape[0]-1)
    image[top, :] = 255 - fill
    image[bottom, :] = 255 - fill
    factor = current_height / float(target_size[1])
    image, _, _ = _do_scale(image, factor)
    return image


def _do_scale(image, factor, max_size=(-1, -1), target_size=(-1, -1)):
    h, w = image.shape[: 2]
    # Calculate new sizes
    ws = max(int(w / factor), 1) if target_size[0] == -1 else target_size[0]
    hs = max(int(h / factor), 1) if target_size[1] == -1 else target_size[1]
    ws = min(ws, max_size[0]) if max_size[0] > -1 else ws
    hs = min(hs, max_size[1]) if max_size[1] > -1 else hs
    if hs != h or ws != w:
        image = cv2.resize(image, (ws, hs))
    return image, ws, hs


def _line_height(image, axis=0):
    n = image.shape[axis]
    m = image.shape[(axis-1) % 2]
    r = np.arange(1, n+1)

    R = as_strided(r, strides=(0, r.itemsize), shape=(m, n))
    if axis == 0:
        R = R.T

    mu = np.average(R, axis=axis, weights=image)
    var = np.average(R**2, axis=axis, weights=image) - mu**2
    std = int(np.average(np.sqrt(var)))
    var = int(np.average(var))
    mu = int(np.average(mu))
    return 2*(int(np.average(std))), mu-std, mu+std
