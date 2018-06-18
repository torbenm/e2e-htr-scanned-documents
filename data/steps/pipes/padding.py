from PIL import Image, ImageOps
import cv2


def pad(imgs, padding=None, fill=255):
    return [_pad_cv2(img, padding, fill) for img in imgs]


def _pad_pil(img, padding=0, fill=255):
    return ImageOps.expand(img, padding, fill=fill)


def _pad_cv2(image, padding=0, fill=255):
    pad_list = padding if isinstance(padding, list) else [padding]*4
    return cv2.copyMakeBorder(image, top=pad_list[0], bottom=pad_list[2], left=pad_list[3], right=pad_list[1], borderType=cv2.BORDER_CONSTANT, value=fill)
