import numpy as np
import cv2

from lib.Configuration import Configuration
from data.steps.pipes import warp, morph, convert, affine, crop, invert, padding, binarize


class ImageAugmenter(object):

    def __init__(self, config):
        self.config = Configuration(config)

    def augment(self, img, get_settings=False):
        augmentation_settings = {}
        if "warp" in self.config["otf_augmentations"]:
            if np.random.uniform() < self.config['otf_augmentations.warp.prob']:
                if(not self.config.default('preprocess.invert', False)):
                    img = 255 - img
                reshaped = False
                if len(img.shape) > 2:
                    reshaped = True
                    img = np.reshape(img, (img.shape[0], img.shape[1]))
                img = convert._cv2pil(img)
                img, mat = warp._warp(
                    img,
                    gridsize=self.config['otf_augmentations.warp.gridsize'],
                    deviation=self.config['otf_augmentations.warp.deviation'],
                    return_mat=True)
                augmentation_settings["warp"] = {
                    "gridsize": self.config['otf_augmentations.warp.gridsize'],
                    "mat": mat
                }
                img = convert._pil2cv2(img)
                if reshaped:
                    img = np.reshape(img, (img.shape[0], img.shape[1], 1))
                if(not self.config.default('preprocess.invert', False)):
                    img = 255 - img
        if "affine" in self.config["otf_augmentations"]:
            if(self.config.default('preprocess.invert', False)):
                img = 255 - img
            img, mat = affine._affine(
                img, self.config["otf_augmentations.affine"], return_mat=True)
            augmentation_settings["affine"] = {
                "mat": mat
            }
            if(self.config.default('preprocess.invert', False)):
                img = 255 - img
        if "morph" in self.config["otf_augmentations"]:
            img, op_name, op_values = morph._random_morph(
                img, self.config["otf_augmentations.morph"], self.config.default('preprocess.invert', False), True)
            augmentation_settings["affine"] = {
                "op_name": op_name,
                "op_values": op_values
            }
        if "binarize" in self.config["otf_augmentations"]:
            if np.random.uniform() < self.config['otf_augmentations.binarize.prob']:
                img = binarize._binarize(img)
                augmentation_settings["binarize"] = {}
        if "blur" in self.config["otf_augmentations"]:
            if np.random.uniform() < self.config['otf_augmentations.blur.prob']:
                img = cv2.GaussianBlur(
                    img, tuple(self.config['otf_augmentations.blur.kernel']), self.config['otf_augmentations.blur.sigma'])
                augmentation_settings["blur"] = {
                    "kernel": self.config['otf_augmentations.blur.kernel'],
                    "sigma": self.config['otf_augmentations.blur.sigma']
                }
        if "sharpen" in self.config["otf_augmentations"]:
            if np.random.uniform() < self.config['otf_augmentations.sharpen.prob']:
                img = self._unsharp_mask_filter(
                    img, tuple(self.config['otf_augmentations.sharpen.kernel']), self.config['otf_augmentations.sharpen.sigma'])
                augmentation_settings["sharpen"] = {
                    "kernel": self.config['otf_augmentations.sharpen.kernel'],
                    "sigma": self.config['otf_augmentations.sharpen.sigma']
                }
        if "brighten" in self.config["otf_augmentations"]:
            if np.random.uniform() < self.config['otf_augmentations.brighten.prob']:
                factor = np.random.normal(
                    self.config['otf_augmentations.brighten.center'], self.config['otf_augmentations.brighten.stdv'])
                factor = factor if factor >= 1 else 1
                img = np.uint8(np.clip(img * factor, 0, 255))
                augmentation_settings["brighten"] = {
                    "factor": factor
                }
        if "darken" in self.config["otf_augmentations"]:
            if np.random.uniform() < self.config['otf_augmentations.darken.prob']:
                factor = np.random.normal(
                    self.config['otf_augmentations.darken.center'], self.config['otf_augmentations.darken.stdv'])
                factor = factor if factor >= 1 else 1
                img = 255 - np.uint8(np.clip((255 - img) * factor, 0.0, 255.0))
                augmentation_settings["darken"] = {
                    "factor": factor
                }
        if not get_settings:
            return self.add_graychannel(img)
        else:
            return self.add_graychannel(img), Configuration(augmentation_settings)

    def binarization(self, img):
        if(self.config.default('preprocess.invert', False)):
            img = 255 - img
        _, img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
        if(self.config.default('preprocess.invert', False)):
            img = 255 - img
        return self.add_graychannel(img)

    def apply_augmentation(self, img, settings):
        if settings.default("warp", False):
            if(not self.config.default('preprocess.invert', False)):
                img = 255 - img
            reshaped = False
            if len(img.shape) > 2:
                reshaped = True
                img = np.reshape(img, (img.shape[0], img.shape[1]))
            img = convert._cv2pil(img)
            img = warp._warp(
                img,
                gridsize=settings['warp.gridsize'],
                mat=settings['warp.mat'])
            img = convert._pil2cv2(img)
            if reshaped:
                img = np.reshape(img, (img.shape[0], img.shape[1], 1))
            if(not self.config.default('preprocess.invert', False)):
                img = 255 - img
        if settings.default("affine", False):
            img = affine._affine(
                img, mat=settings["affine.mat"], background=255.0)
        if settings.default("morph", False):
            img = morph._morph(img, settings['morph.op_name'], settings['morph.op_values'], self.config.default(
                'preprocess.invert', False))
        if settings.default("binarize", False):
            img = binarize._binarize(img)
        if settings.default("blur", False):
            img = cv2.GaussianBlur(
                img, tuple(settings['blur.kernel']), settings['blur.sigma'])
        if settings.default("sharpen", False):
            img = self._unsharp_mask_filter(
                img, tuple(settings['sharpen.kernel']), settings['sharpen.sigma'])
        if settings.default("brighten", False):
            img = np.uint8(
                np.clip(img * settings["brighten.factor"], 0.0, 255.0))
        if settings.default("darken", False):
            img = 255 - np.uint8(
                np.clip((255 - img) * settings["darken.factor"], 0.0, 255.0))
        return self.add_graychannel(img)

    def _unsharp_mask_filter(self, image, kernel, sigma):
        gaussian_3 = cv2.GaussianBlur(image, kernel, sigma)
        return cv2.addWeighted(image, 1.5, gaussian_3, -0.5, 0, image)

    def add_graychannel(self, img):
        if len(img.shape) == 2:
            return np.reshape(img, [img.shape[0], img.shape[1], 1])
        return img

    def pad_to_size(self, img, height, width):
        return self._pad(img, (height, width, 1))

    def _scale(self, img, factor, target_size=None):
        height = int(img.shape[0] / factor)
        width = int(img.shape[1] / factor)
        if width <= 0 or height <= 0:
            return None
        return cv2.resize(img, (width, height))

    def _scale_img(self, img, scale_factor, target_size=None):
        if img.shape[0] == 0 or img.shape[1] == 0:
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
            if img.shape[0] == 0 or img.shape[1] == 0:
                return None
            img = crop._crop(img)
            if img is None:
                return None

        if self.config.default('preprocess.scale', False):
            img = self._scale_img(
                img, self.config['preprocess.scale'], target_size)
            if img is None:
                return None

        if self.config.default('preprocess.padding', False):
            img = padding._pad_cv2(img, self.config['preprocess.padding'], bg)
        img = self.add_graychannel(img)
        if target_size != None:
            target_size = (
                target_size[0] +
                (self.config.default('preprocess.padding', 0)*2),
                target_size[1] +
                (self.config.default('preprocess.padding', 0)*2),
                1
            )
            img = self._pad(img, target_size)
        return img

    def postprocesss(self, img):
        if self.config.default('postprocess.binarize', False):
            img = self.binarization(img)
        return img

    def _pad(self, array, reference_shape, offsets=None):
        """
        array: Array to be padded
        reference_shape: tuple of size of ndarray to create
        offsets: list of offsets (number of elements must be equal to the dimension of the array)
        will throw a ValueError if offsets is too big and the reference_shape cannot handle the offsets
        """
        offsets = offsets if offsets is not None else [
            0] * len(array.shape)
        # Create an array of zeros with the reference shape
        result = np.zeros(reference_shape)
        # Create a list of slices from offset to offset + shape in each dimension
        insertHere = [slice(offsets[dim], offsets[dim] + array.shape[dim])
                      for dim in range(array.ndim)]
        # Insert the array in the result at the specified offsets
        result[tuple(insertHere)] = array
        return result
