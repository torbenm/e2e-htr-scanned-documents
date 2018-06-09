from PIL import Image, ImageOps
import cv2


def pad(imgs, padding=None, fill=(255, 255, 255)):
    if padding is not None and padding > 0:
        return [_pad_cv2(img, padding, fill) for img in imgs]
    return imgs


def _pad_pil(img, padding=0, fill=(255, 255, 255)):
    return ImageOps.expand(img, padding, fill=fill)


def _pad_cv2(image, padding=0, fill=(255, 255, 255)):
    return cv2.copyMakeBorder(image, top=padding, bottom=padding, left=padding, right=padding, borderType=cv2.BORDER_CONSTANT, value=fill)
