import cv2


def getScaleFactor(imagepaths, target_size, hook=None):
    factor = 0
    i = 0
    l = len(imagepaths)
    hookstep = l // 100
    hookstep = hookstep if hookstep > 0 else 1

    def make_factor(target):
        def _nofactor(val):
            return 0

        def _factor(value):
            return float(value) / target
        return _factor if target != -1 else _nofactor

    w_factor = make_factor(target_size[0])
    h_factor = make_factor(target_size[1])

    for ipath in imagepaths:
        image = cv2.imread(ipath)
        i = i + 1
        if image is not None:
            h, w, _ = image.shape
            factor = max(h_factor(h), w_factor(w), factor)
        if hook is not None and i % hookstep == 0:
            hook(i, l)
    return factor
