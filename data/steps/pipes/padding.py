from PIL import Image, ImageOps


def pad(imgs, padding=None, fill='white'):
    if padding is not None:
        return [_pad(img, padding, fill) for img in imgs]
    return imgs


def _pad(img, padding=0, fill='white'):
    return ImageOps.expand(img, padding, fill=fill)
