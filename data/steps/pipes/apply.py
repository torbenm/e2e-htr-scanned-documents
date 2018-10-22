from PIL import Image
import cv2
import numpy as np
import os
from .padding import pad
from .invert import invert
from .warp import RandomWarpGridDistortion
from .binarize import binarize
from .threshold import threshold
from .scale import scale
from .save import save
from .convert import pil2cv2, cv2pil
from .grayscale import toGrayscale
from .crop import crop
from .morph import morph
from .copy import copy
from .affine import affine_but_first


def applyPipeline(sourcepath, truth, type, context, train):

    def isActive(prop, ctx=context,  default=False):
        if prop in ctx:
            return bool(ctx[prop])
        return default

    bgColor = 255

    # Step 0: Read Image as Grayscale
    images = [cv2.imread(sourcepath, cv2.IMREAD_GRAYSCALE)]
    if images[0] is None or images[0].shape[0] == 0 or images[0].shape[1] == 0:
        return [], (0, 0)

    # Step 1: Invert Image
    if isActive('invert'):
        images = invert(images)
        bgColor = 255 - bgColor

    # Step 2: Apply treshold, if wanted
    if isActive('threshold'):
        images = threshold(images, invert=isActive('invert'))

    # Step 3: Crop Image
    if isActive('crop'):
        images = crop(images)

    if images[0] is None or images[0].shape[0] == 0 or images[0].shape[1] == 0:
        return [], (0, 0)

    # Step 7: Warp Image
    if train and isActive('warp') and isActive('num', ctx=context['warp']):
        images = cv2pil(images)
        images = RandomWarpGridDistortion(
            images, context['warp']['num'], context['warp']['gridsize'], context['warp']['deviation'])
        images = pil2cv2(images)
    elif train and isActive('copy'):
        images = copy(images, context['copy'])

    if train and isActive('morph'):
        images = morph(images, context['morph'], invert=isActive('invert'))

    if train and isActive('affine'):
        images = affine_but_first(images, context['affine'])

    # Step 4: Scale Image
    if isActive('scale'):
        images = scale(images, context['scale'], bgColor)

    # Step 5: Add padding
    if isActive('padding'):
        images = pad(images, context['padding'], fill=bgColor)

    # Step 6: Extract width & height
    h, w = images[0].shape[:2]

    # Step 8: Binarize Image
    if isActive('binarize'):
        images = binarize(images)

    # Step 9: Save Images
    imagepaths = save(images, splitext(sourcepath), context['imagetarget'])

    return [{"truth": truth, "path": imagepath, "type": type} for imagepath in imagepaths], (w, h)


def splitext(path):
    return os.path.splitext(os.path.basename(path))
