from pipes import apply


def applyFullPipeline(images, context, hook=None, train=False):
    fullset = []
    i = 0
    l = len(images)
    hookstep = l // 100
    hookstep = hookstep if hookstep > 0 else 1
    for image in images:
        fullset.extend(apply.applyPipeline(
            image['path'], image['truth'], context, train))
        if hook is not None and i % hookstep == 0:
            hook(i, l)
        i = i + 1
    return fullset
