from PIL import Image, ImageOps


def invert(imgs):
    return [ImageOps.invert(img) for img in imgs]
