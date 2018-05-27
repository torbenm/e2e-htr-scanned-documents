def makeIndex(context):
    if 'scale' in context:
        indexContent = {
            "width": context['scale']['size'][0] * context['scale']['factor'],
            "height": context['scale']['size'][1] * context['scale']['factor']
        }
    else:
        indexContent = {}
    return indexContent
