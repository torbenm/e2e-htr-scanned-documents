from PIL import ImageOps


def toGrayscale(imgs):
    return [ImageOps.grayscale(img) for img in imgs]
