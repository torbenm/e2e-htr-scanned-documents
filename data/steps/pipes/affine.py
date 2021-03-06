import cv2
import numpy as np
from lib.Configuration import Configuration


def affine(images, config):
    return [_affine(image, config) for image in images]


def affine_but_first(images, config):
    images_all = [images[0]]
    altered = affine(images[1:], config)
    images_all.extend(altered)
    return images_all


def _affine(image, config=None, mat=None, background=0, return_mat=False):
    at = AffineTransformation(image)
    if config is not None:
        at.configure(config)
    if mat is not None:
        at.set(mat)
    if return_mat:
        return at(background=background), at.M
    else:
        return at(background=background)


class AffineTransformation(object):

    DEFAULTS = Configuration({
        "translate": {
            "prob": 0.5,
            "center": 2,
            "stdv": 0.02
        },
        "rotate": {
            "prob": 0.5,
            "center": 0,
            "stdv": 0.1
        },
        "shear": {
            "prob": 0.5,
            "center": 0,
            "stdv": 0.25
        },
        "scale": {
            "prob": 0.5,
            "center": 0,
            "stdv": 0.06
        }
    })

    def __init__(self, img):
        self.reset()
        self.img = img

    def reset(self):
        self.M = np.eye(3, 3)

    def set(self, M):
        self.M = M

    def __call__(self, background=None):
        if background is None:
            return cv2.warpAffine(self.img, self.M[:2, :], (self.img.shape[1], self.img.shape[0]))
        else:
            return cv2.warpAffine(self.img, self.M[:2, :], (self.img.shape[1], self.img.shape[0]), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

    ######################################
    # Random Transformations            #
    #####################################
    def translate(self, prob=0.5, stdv=0.02, center=2):
        if np.random.uniform() < prob:
            t = np.random.normal(center, stdv)
            self._apply(self._translate(t, t))

    def rotate(self, prob=0.5, stdv=0.2, center=0):
        if np.random.uniform() < prob:
            stdv = np.sqrt(1/max(self.img.shape[0] / self.img.shape[1],
                                 self.img.shape[1] / self.img.shape[0])) * stdv
            alpha = np.random.normal(center, stdv)
            self._centered(self._rotate(alpha))

    def shear(self, prob=0.5, stdv=0.25, center=0):
        if np.random.uniform() < prob:
            s = np.random.normal(center, stdv)
            self._centered(self._shear(s, 0))

    def scale(self, prob=0.5, stdv=0.12, center=0):
        if np.random.uniform() < prob:
            f = np.exp(np.random.normal(center, stdv))
            self._centered(self._scale(f, f))

    def configure(self, config={}):
        config = Configuration(config)

        def conf(key):
            return config.default(
                key, self.DEFAULTS.default(key, 1))

        self.translate(prob=conf('translate.prob'),
                       center=conf('translate.center'),
                       stdv=conf('translate.stdv'))
        self.rotate(prob=conf('rotate.prob'),
                    center=conf('rotate.center'),
                    stdv=conf('rotate.stdv'))
        self.shear(prob=conf('shear.prob'),
                   center=conf('shear.center'),
                   stdv=conf('shear.stdv'))
        self.scale(prob=conf('scale.prob'),
                   center=conf('scale.center'),
                   stdv=conf('scale.stdv'))

        ######################################
        # Tranformation Matrix Builder       #
        ######################################

    def _apply(self, D):
        self.M = np.matmul(self.M, D)

    def _centered(self, D):
        tx = self.img.shape[1] / 2
        ty = self.img.shape[0] / 2
        C = self._translate(tx, ty)
        Cm = self._translate(-tx, -ty)
        self.M = np.matmul(np.matmul(C, D), Cm)

    def _rotate(self, alpha):
        alpha = np.deg2rad(alpha)
        return np.float32([
            [np.cos(alpha), np.sin(alpha), 0],
            [-np.sin(alpha), np.cos(alpha), 0],
            [0, 0, 1]
        ])

    def _translate(self, tx, ty):
        return np.float32([
            [1, 0, tx],
            [0, 1, ty],
            [0, 0, 1]
        ])

    def _scale(self, fx, fy):
        return np.float32([
            [fx, 0, 0],
            [0, fy, 0],
            [0, 0, 1]
        ])

    def _shear(self, sx, sy):
        return np.float32([
            [1, sx, 0],
            [sy, 1, 0],
            [0, 0, 1]
        ])
