from util import apply


def invert(images, config, context):
    context['inverted'] = True
    apply(_invert, images, config, context)


def _invert(image, config, context):
    return 255 - image
