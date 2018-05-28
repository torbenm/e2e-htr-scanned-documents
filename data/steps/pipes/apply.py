from PIL import Image
import cv2
import numpy as np
import os
from padding import pad
from invert import invert
from warp import RandomWarpGridDistortion
from binarize import binarize
from scale import scale
from save import save


def applyPipeline(sourcepath, truth, context, train):
    bgColor = (255, 255, 255)
    # The pipe works with list, so load the image into a list
    images = [Image.open(sourcepath)]

    # Step 1: Apply padding. If no padding is defined, none is applied.
    images = pad(images, context['padding'])

    # Step 2: Invert colors.
    if context['invert']:
        images = invert(images)
        bgColor = (0, 0, 0)

    # Step 3: Create warped variants
    if train and 'warp' in context and 'num' in context['warp'] and context['warp']['num'] > 0:
        images = RandomWarpGridDistortion(
            images, context['warp']['num'], context['warp']['gridsize'], context['warp']['deviation'])
    # The following steps are implemented using cv2 - therefore we need
    # conversion
    images = pil2cv2(images)

    # Step 4: Scale
    if 'scale' in context:
        images = scale(images, context['scale'][
                       'factor'], context['scale']['size'], bgColor, context['padding'])

    # Step 5: Binarize
    if context['binarize']:
        images = binarize(images)

    # Step 6: Save
    imagepaths = save(images, splitext(sourcepath), context['imagetarget'])

    return [{"truth": truth, "path": imagepath} for imagepath in imagepaths]


def splitext(path):
    return os.path.splitext(os.path.basename(path))


def pil2cv2(images):
    return [_pil2cv2(image) for image in images]


def _pil2cv2(image):
    npa = np.array(image.convert('RGB'))
    return npa[:, :, ::-1].copy()
