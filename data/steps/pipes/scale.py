import cv2


def scale(images, scale=1.0, size=(100, 100), fill=(255, 255, 255), padding=None):
    return [_scale(image, scale, size, fill, padding) for image in images]


def _scale(image, scale=1.0, size=(100, 100), fill=(255, 255, 255), padding=None):
    width, height = size
    if padding is not None:
        width = int(width + padding * scale)
        height = int(height + padding * scale)
    h, w = image.shape[:2]
    ws = min(max(int(w / scale), 1), width)
    hs = min(max(int(h / scale), 1), height)
    if scale != 1.0:
        image = cv2.resize(image, (ws, hs))
    return cv2.copyMakeBorder(
        image, 0, height - hs, 0, width - ws, cv2.BORDER_CONSTANT, value=fill)
