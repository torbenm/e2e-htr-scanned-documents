import cv2

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
    return cv2.copyMakeBorder(
        image, 0, height - hs, 0, width - ws, cv2.BORDER_CONSTANT, value=fill)


def _scale_by_height(image, config, fill=(255, 255, 255)):
    target_size = config['size']
    current_height = image.shape[0]
    factor = current_height / target_size[1]
    image, _, _ = _do_scale(image, factor, target_size=target_size)
    return image


def _do_scale(image, factor, max_size=(-1, -1), target_size=(-1, -1)):
    h, w = image.shape[:2]
    # Calculate new sizes
    ws = max(int(w / factor), 1) if target_size[0] == -1 else target_size[0]
    hs = max(int(h / factor), 1) if target_size[1] == -1 else target_size[1]
    ws = min(ws, max_size[0]) if max_size[0] > -1 else ws
    hs = min(hs, max_size[1]) if max_size[1] > -1 else hs
    if hs != h or ws != w:
        image = cv2.resize(image, (ws, hs))
    return image, ws, hs
