def makeIndex(context):
    if 'scale' in context:
        metaContent = {
            "width": context['scale']['size'][0],
            "height": context['scale']['size'][1]
        }
        if 'padding' in context:
            metaContent["width"] = int(metaContent["width"] +
                                       context['padding'] * context['scale']['factor'])
            metaContent["height"] = int(metaContent["height"] +
                                        context['padding'] * context['scale']['factor'])
    else:
        metaContent = {}
    return metaContent
