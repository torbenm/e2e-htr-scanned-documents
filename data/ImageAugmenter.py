import numpy as np
import cv2

from lib.Configuration import Configuration
from data.steps.pipes import warp, morph, convert, affine, crop, invert, padding


class ImageAugmenter(object):

    def __init__(self, config):
        self.config = Configuration(config)

    def augment(self, img):
        if "warp" in self.config["otf_augmentations"]:
            if np.random.uniform() < self.config['otf_augmentations.warp.prob']:
                img = convert._cv2pil(img)
                img = warp._warp(
                    img, gridsize=self.config['otf_augmentations.warp.gridsize'], deviation=self.config['otf_augmentations.warp.deviation'])
                img = convert._pil2cv2(img)
        if "affine" in self.config["otf_augmentations"]:
            img = affine._affine(
                img, self.config["otf_augmentations.affine"])
        if "morph" in self.config["otf_augmentations"]:
            img = morph._random_morph(
                img, self.config["otf_augmentations.morph"], True)
        return img

    def add_graychannel(self, img):
        return np.reshape(img, [img.shape[0], img.shape[1], 1])

    def pad_to_size(self, img, height, width):
        return self._pad(img, (height, width))

    def _scale(self, img, factor, target_size=None):
        height = int(img.shape[0] / factor)
        width = int(img.shape[1] / factor)
        if width <= 0 or height <= 0:
            print('w and h is <= 0', width, height)
            return None
        return cv2.resize(img, (width, height))

    def _scale_img(self, img, scale_factor, target_size=None):
        if img.shape[0] == 0 or img.shape[1] == 0:
            print('prescale shape is 0', img.shape)
            return None
        factor = max(img.shape[0] / target_size[0],
                     img.shape[1] / target_size[1],
                     scale_factor)
        img = self._scale(img, factor)
        return img

    def preprocess(self, img, target_size=None):
        bg = 255
        if self.config.default('preprocess.invert', False):
            img = invert._invert(img)
            bg = 255 - bg

        if self.config.default('preprocess.crop', False):
            print('precrop shape is 0', img.shape)
            if img.shape[0] == 0 or img.shape[1] == 0:
                return None
            img = crop._crop(img)

        if self.config.default('preprocess.scale', False):
            img = self._scale_img(
                img, self.config['preprocess.scale'], target_size)
            if img is None:
                return None

        if self.config.default('preprocess.padding', False):
            img = padding._pad_cv2(img, self.config['preprocess.padding'], bg)

        if target_size != None:
            target_size = (
                target_size[0] +
                (self.config.default('preprocess.padding', 0)*2),
                target_size[1] +
                (self.config.default('preprocess.padding', 0)*2)
            )
            img = self._pad(img, target_size)
        return img

    def _pad(self, array, reference_shape, offsets=None):
        """
        array: Array to be padded
        reference_shape: tuple of size of ndarray to create
        offsets: list of offsets (number of elements must be equal to the dimension of the array)
        will throw a ValueError if offsets is too big and the reference_shape cannot handle the offsets
        """
        offsets = offsets if offsets is not None else [
            0] * len(reference_shape)
        # Create an array of zeros with the reference shape
        result = np.zeros(reference_shape)
        # Create a list of slices from offset to offset + shape in each dimension
        insertHere = [slice(offsets[dim], offsets[dim] + array.shape[dim])
                      for dim in range(array.ndim)]
        # Insert the array in the result at the specified offsets
        result[insertHere] = array
        return result
