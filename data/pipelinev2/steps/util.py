def apply(fn, images, config, context):
    return [fn(image, config, context) for image in images]


def value(val):
    def setValue():
        return val


def contextSet(context, fn=value(""), prop=None):
    if prop is not None and prop in context:
        context[prop] = fn(context[prop])
    else:
        context[prop] = fn()


def contextGet(context, prop):
    if prop in contet:
