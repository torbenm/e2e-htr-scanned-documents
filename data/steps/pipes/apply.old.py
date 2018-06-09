from PIL import Image
import cv2
import numpy as np
import os
from padding import pad
from invert import invert
from warp import RandomWarpGridDistortion
from binarize import binarize
from threshold import threshold
from scale import scale
from save import save
from convert impot pil2cv2
from grayscale import toGrayscale


def applyPipeline(sourcepath, truth, context, train):
    bgColor = np.asarray((255, 255, 255))
    isInverted = False
    # The pipe works with list, so load the image into a list
    images = [Image.open(sourcepath)]

    # Step 0.1: To Grayscale, if necessary
    if images[0].mode != "L":
        images = toGrayscale(images)

    # Step 1: Apply padding. If no padding is defined, none is applied.
    images = pad(images, context['padding'], fill=bgColor)

    # Step 2: Invert colors.
    if 'invert' in context and context['invert']:
        images = invert(images)
        bgColor = 255 - bgColor
        isInverted = True

    # Step 3: Create warped variants
    if train and 'warp' in context and 'num' in context['warp'] and context['warp']['num'] > 0:
        images = RandomWarpGridDistortion(
            images, context['warp']['num'], context['warp']['gridsize'], context['warp']['deviation'])
    # The following steps are implemented using cv2 - therefore we need
    # conversion
    images = pil2cv2(images)

    if 'threshold' in context and ['threshold']:
        images = threshold(images, invert=isInverted)

    # Step 4: Scale
    if 'scale' in context:
        images = scale(images, context['scale'][
                       'factor'], context['scale']['size'], bgColor, context['padding'])

    # Step 5: Binarize or threshold
    if 'binarize' in context and context['binarize']:
        images = binarize(images)

    # Step 6: Save
    imagepaths = save(images, splitext(sourcepath), context['imagetarget'])

    return [{"truth": truth, "path": imagepath} for imagepath in imagepaths]


def splitext(path):
    return os.path.splitext(os.path.basename(path))
