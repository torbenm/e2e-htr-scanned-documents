def makeIndex(context, sizes=None):
    if sizes is None and 'scale' in context:
        metaContent = {
            "width": context['scale']['size'][0],
            "height": context['scale']['size'][1]
        }
        if 'padding' in context:
            metaContent["width"] = int(metaContent["width"] +
                                       context['padding'] * context['scale']['factor'])
            metaContent["height"] = int(metaContent["height"] +
                                        context['padding'] * context['scale']['factor'])
    elif sizes is not None:
        metaContent = {
            "width": int(sizes[0]),
            "height": int(sizes[1])
        }
    else:
        metaContent = {}
    return metaContent
