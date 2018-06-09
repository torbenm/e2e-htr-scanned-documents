def invert(images, maxVal=255):
    return [maxVal - image for image in images]
