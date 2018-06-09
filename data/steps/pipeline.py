from pipes import apply


def applyFullPipeline(images, context, hook=None, train=False):
    fullset = []
    i = 0
    l = len(images)
    hookstep = l // 100
    hookstep = hookstep if hookstep > 0 else 1
    max_w, max_h = 0, 0
    for image in images:

        prepared_image, size = (apply.applyPipeline(
            image['path'], image['truth'], context, train))
        max_w = max(size[0], max_w)
        max_h = max(size[1], max_h)
        fullset.extend(prepared_image)
        if hook is not None and i % hookstep == 0:
            hook(i, l)
        i = i + 1
    return fullset, (max_w, max_h)
