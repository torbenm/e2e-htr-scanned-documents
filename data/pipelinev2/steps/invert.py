from util import apply


def invert(images, config, context):
    if 'bgColor' in context:
    apply(_invert, images, config, context)


def _invert(image, config, context):
    return 255 - image
