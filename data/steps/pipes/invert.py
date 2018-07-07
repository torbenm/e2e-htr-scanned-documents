def invert(images, maxVal=255):
    return [_invert(image, maxVal) for image in images]


def _invert(image, maxVal=255):
    return maxVal - image
